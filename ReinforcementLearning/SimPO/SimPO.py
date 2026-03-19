import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer

# ==========================================
# 1. 加载模型与 Tokenizer (注意：这里不需要 ref_model 了！显存省一半)
# ==========================================
config = LlamaConfig(vocab_size=32000, hidden_size=256, num_hidden_layers=2)
model = LlamaForCausalLM(config) # 只有一个主力模型

my_cache_path = "../LLMS"
tokenizer = AutoTokenizer.from_pretrained(
    'HuggingFaceM4/tiny-random-LlamaForCausalLM',
    cache_dir=my_cache_path
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id or 0

# ==========================================
# 2. 准备 SimPO 数据集 (基于你的 KTO 数据扩充为成对格式)
# ==========================================
simpo_dataset = {
    "prompt": [
        "How are you", 
        "What is your name?", 
        "Which is the best programming language?",
        "How to kill a man?"
    ],
    "chosen": [
        "hi nice to meet you",                  # 原 KTO 的 True
        "I am a helpful AI assistant.",         # 为原 False 补充的好回答
        "Python is great for AI.",              # 为原 True 扩写，体现长度归一化优势
        "I cannot fulfill this request."        # 为原 False 补充的安全回答
    ],
    "rejected": [
        "I don't care.",                        # 为原 True 补充的坏回答
        "leave me alone",                       # 原 KTO 的 False
        "Python",                               # 原 KTO 的 True (超短回答，用来测试长度惩罚)
        "Use gun shoot his head"                # 原 KTO 的 False
    ]
}

# ==========================================
# 3. 数据处理与辅助函数
# ==========================================
def get_logprobs_and_lengths(model, tokenizer, prompts, completions):
    """
    辅助函数：拼接文本、过模型、提取回答部分的 Log 概率和长度
    """
    logprobs_list = []
    lengths_list = []
    
    for prompt, comp in zip(prompts, completions):
        # 编码
        prompt_ids = tokenizer('USER:' + prompt + 'ASSISTANT:', add_special_tokens=True)['input_ids']
        comp_ids = tokenizer(comp, add_special_tokens=False)['input_ids']
        
        input_ids = torch.tensor([prompt_ids + comp_ids])
        mask = torch.tensor([[0] * len(prompt_ids) + [1] * len(comp_ids)]) # 只计算回答部分的 Loss
        
        # 前向传播 (不用计算梯度以节省显存，实际训练时这里会有 loss.backward())
        logits = model(input_ids).logits
        
        # 提取 log 概率
        log_probs = logits.log_softmax(-1)
        per_token_logps = torch.gather(log_probs, dim=2, index=input_ids.unsqueeze(2)).squeeze(2)
        
        # 过滤 prompt 部分并求和，得到该回答的 "总 Log 概率"
        answer_logp_sum = (per_token_logps * mask).sum()
        answer_length = mask.sum()
        
        logprobs_list.append(answer_logp_sum)
        lengths_list.append(answer_length)
        
    return torch.stack(logprobs_list), torch.stack(lengths_list)

# 提取 Chosen 和 Rejected 的总概率与长度
# 现实中通常会对齐成 Batch 并行计算，这里为了逻辑清晰使用 for 循环提取
chosen_logps, chosen_lengths = get_logprobs_and_lengths(model, tokenizer, simpo_dataset["prompt"], simpo_dataset["chosen"])
rejected_logps, rejected_lengths = get_logprobs_and_lengths(model, tokenizer, simpo_dataset["prompt"], simpo_dataset["rejected"])

# ==========================================
# 4. SimPO 核心 Loss 计算
# ==========================================
# 超参数设置
beta = 2.0   # SimPO 的 beta 通常较大 (2.0 - 2.5)
gamma = 0.5  # 目标间隔 Margin

# 步骤 A: 长度归一化 (核心！防止模型刷废话)
# 用 "总分" 除以 "词数"，得到 "每个 Token 的平均分"
avg_logp_chosen = chosen_logps / chosen_lengths
avg_logp_rejected = rejected_logps / rejected_lengths

# 步骤 B: 计算隐式奖励
reward_chosen = beta * avg_logp_chosen
reward_rejected = beta * avg_logp_rejected

# 步骤 C: 计算差值并扣除 Margin
# 我们希望好坏奖励的差值不仅大于 0，还要大于 gamma
logits_diff = reward_chosen - reward_rejected - gamma

# 步骤 D: 计算最终 Loss (Bradley-Terry 目标)
# 等价于 -log(sigmoid(logits_diff))
simpo_loss = -F.logsigmoid(logits_diff).mean()

print(f"--- SimPO 计算结果 ---")
print(f"好回答平均奖励 (Reward Chosen): {reward_chosen.detach().numpy()}")
print(f"坏回答平均奖励 (Reward Rejected): {reward_rejected.detach().numpy()}")
print(f"最终 Batch Loss: {simpo_loss.item():.4f}")