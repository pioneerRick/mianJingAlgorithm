import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
from datasets import Dataset

# 1. 加载模型与参考模型 (使用小型配置做演示)
config = LlamaConfig(vocab_size=32000, hidden_size=256, num_hidden_layers=2)
model = LlamaForCausalLM(config)
ref_model = LlamaForCausalLM(config)
ref_model.eval() # 参考模型冻结，不参与梯度更新

# 2. 准备 KTO 数据集 (非配对数据，只有 True/False 标签)
kto_dataset_dict = {
    "prompt": [
        "How are you", 
        "What is your name?", 
        "Which is the best programming language?",
        "How to kill a man?"
    ],
    "completion": [
        "hi nice to meet you", 
        "leave me alone", 
        "Python", 
        "Use gun shoot his head"
    ],
    "label": [True, False, True, False] # True 为好回答，False 为坏回答
}
dataset_raw = Dataset.from_dict(kto_dataset_dict)
my_cache_path = "../LLMS"

tokenizer = AutoTokenizer.from_pretrained(
    'HuggingFaceM4/tiny-random-LlamaForCausalLM',
    cache_dir=my_cache_path  # 添加这行
)

# 3. 数据处理：拼接 Prompt 和 Completion，并生成 Mask
def process_kto_dataset(example):
    prompt_id = tokenizer('USER:' + example['prompt'] + 'ASSISTANT:', add_special_tokens=True)['input_ids']
    completion_id = tokenizer(example['completion'], add_special_tokens=False)['input_ids']
    
    example['input_ids'] = prompt_id + completion_id
    # label_mask: 0 表示不计算 Loss (Prompt部分)，1 表示计算 Loss (回答部分)
    example['label_mask'] = [0] * len(prompt_id) + [1] * len(completion_id)
    example['attention_mask'] = [1] * len(example['input_ids'])
    return example

dataset = dataset_raw.map(process_kto_dataset, batched=False)

# print(dataset.shape) # (4, 6)

# 1. 确保 tokenizer 有 pad_token (LLaMA 默认可能没有，借用 eos_token)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id or 0
pad_id = tokenizer.pad_token_id

# 2. 找到当前 batch 中的最大长度
max_length = max(len(ids) for ids in dataset['input_ids'])

# 3. 对所有序列进行填充补齐
padded_input_ids = []
padded_attention_mask = []
padded_label_mask = []

for ids, a_mask, l_mask in zip(dataset['input_ids'], dataset['attention_mask'], dataset['label_mask']):
    # 计算需要补多少个 token
    pad_len = max_length - len(ids)
    
    # 在末尾补齐
    padded_input_ids.append(ids + [pad_id] * pad_len)
    padded_attention_mask.append(a_mask + [0] * pad_len) # Attention 忽略 pad 部分，补 0
    padded_label_mask.append(l_mask + [0] * pad_len)     # Loss 忽略 pad 部分，补 0

# 4. 现在长度都一致了，可以安全地转换为 Tensor
batch_input_ids = torch.tensor(padded_input_ids)
batch_attention_mask = torch.tensor(padded_attention_mask)
batch_label_mask = torch.tensor(padded_label_mask)
labels = torch.tensor(dataset['label'])

# 4. 获取当前模型和参考模型的 Logits
logits = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits
with torch.no_grad():
    ref_logits = ref_model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits

# 5. 提取每个 Token 的对数概率 (Log Probabilities)
def get_probs(logits, input_ids, mask):
    # 计算 log softmax
    log_probs = logits.log_softmax(-1)
    # 获取实际 token 对应的概率
    per_token_logps = torch.gather(log_probs, dim=2, index=input_ids.unsqueeze(2)).squeeze(2)
    # 利用 mask 过滤掉 Prompt 部分，仅保留 Completion 部分并求平均
    per_token_logps = (per_token_logps * mask).sum(-1) / mask.sum(-1)
    return per_token_logps

probs = get_probs(logits, batch_input_ids, batch_label_mask)
ref_probs = get_probs(ref_logits, batch_input_ids, batch_label_mask)

# 6. 计算 KL 散度基准 (整个 batch 的平均概率差异)
kl = (probs - ref_probs).mean().detach()

# 7. 区分好回答 (Chosen) 和坏回答 (Rejected)
chosen_idx = torch.where(labels == True)[0]
rejected_idx = torch.where(labels == False)[0]



chosen_probs = probs[chosen_idx]
ref_chosen_probs = ref_probs[chosen_idx]

rejected_probs = probs[rejected_idx]
ref_rejected_probs = ref_probs[rejected_idx]

# print(rejected_probs.shape) # torch.Size([2])

# 8. 计算概率比 (Ratio)
chosen_ratio = chosen_probs - ref_chosen_probs
rejected_ratio = rejected_probs - ref_rejected_probs

# 9. KTO 超参数设置
beta = 0.1
desirable_weight = 1.33
undesirable_weight = 1.0

# 10. 计算损失 (应用 Sigmoid 和权重)
# 好回答：我们希望 chosen_ratio 越大越好
chosen_losses = 1 - F.sigmoid(beta * (chosen_ratio - kl))

# 坏回答：我们希望 rejected_ratio 越小越好 (注意这里的符号逻辑，根据图8实现)
# 注：标准的 KTO 公式对 rejected_loss 是 sigmoid(beta * (kl - rejected_ratio))
# 你图片中的代码写的是 beta * (rejected_ratio - kl)，这取决于具体代码库中 reward 的正负定义。
rejected_losses = 1 - F.sigmoid(beta * (rejected_ratio - kl)) 

# 合并计算平均 Loss，准备进行反向传播更新模型
losses = torch.cat([
    desirable_weight * chosen_losses, 
    undesirable_weight * rejected_losses
])
kto_loss = losses.nanmean()

print(f"Final KTO Loss: {kto_loss.item()}")
# kto_loss.backward() -> optimizer.step() ...