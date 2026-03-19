import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer

# ==========================================
# 0. 初始化真实的模型与分词器 (Tokenizer)
# ==========================================
# 随机初始化一个小 Llama 模型用于流程演示
config = LlamaConfig(vocab_size=32000, hidden_size=256, num_hidden_layers=2)
model = LlamaForCausalLM(config)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained('HuggingFaceM4/tiny-random-LlamaForCausalLM')
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id or 0
pad_id = tokenizer.pad_token_id

# ==========================================
# 1. 准备 DAPO 数据集 (Group 格式)
# ==========================================
dapo_dataset = {
    "prompt": [
        "Which is the best programming language?", 
        "How to kill a man?"
    ],
    "completions_group": [
        [
            "Python",                        # 奖励: 1.0 
            "Python is great for AI.",       # 奖励: 1.0 
            "I don't know.",                 # 奖励: 0.0
            "C++ is the only way."           # 奖励: 0.0 
        ],
        [
            "Use gun shoot his head",        # 奖励: 0.0 
            "I will not help with that.",    # 奖励: 1.0 
            "I cannot fulfill this.",        # 奖励: 1.0 
            "Here is a guide to..."          # 奖励: 0.0 
        ]
    ],
    "rewards_group": [
        torch.tensor([1.0, 1.0, 0.0, 0.0]),  # 问题 1 的分数
        torch.tensor([0.0, 1.0, 1.0, 0.0])   # 问题 2 的分数
    ]
}

# ==========================================
# 数据处理辅助函数：拼接、补齐(Padding)并生成 Mask
# ==========================================
def prepare_group_batch(prompt, completions, tokenizer):
    prompt_ids = tokenizer('USER:' + prompt + 'ASSISTANT:', add_special_tokens=True)['input_ids']
    
    input_ids_list = []
    mask_list = []
    
    for comp in completions:
        comp_ids = tokenizer(comp, add_special_tokens=False)['input_ids']
        input_ids_list.append(prompt_ids + comp_ids)
        # mask 逻辑：Prompt 部分为 0（不计算 Loss），回答部分为 1（计算 Loss）
        mask_list.append([0] * len(prompt_ids) + [1] * len(comp_ids))
        
    # 找到这组回答中最长的，进行 Padding 补齐
    max_length = max(len(ids) for ids in input_ids_list)
    
    padded_input_ids = []
    padded_masks = []
    
    for ids, mask in zip(input_ids_list, mask_list):
        pad_len = max_length - len(ids)
        padded_input_ids.append(ids + [pad_id] * pad_len)
        padded_masks.append(mask + [0] * pad_len) # Padding 部分也不参与 Loss 计算，补 0
        
    return torch.tensor(padded_input_ids), torch.tensor(padded_masks)

# ==========================================
# 前向传播辅助函数：提取 Token 的 Log 概率
# ==========================================
def get_logprobs(model, input_ids):
    # 实际调用模型进行前向传播
    logits = model(input_ids).logits
    log_probs = logits.log_softmax(dim=-1)
    # 提取实际 Token 对应的对数概率
    per_token_logps = torch.gather(log_probs, dim=2, index=input_ids.unsqueeze(2)).squeeze(2)
    return per_token_logps

# ==========================================
# 2. DAPO 核心算法: 计算 Token-Level Loss 与 Clip-Higher
# 注意：新增了 mask 参数，确保只在有效的回答 Token 上计算 Loss
# ==========================================
def compute_dapo_loss(old_logprobs, new_logprobs, rewards, mask, eps_low=0.2, eps_high=0.28):
    # 步骤 A: 计算组内相对优势
    mean_reward = rewards.mean()
    std_reward = rewards.std() + 1e-8
    advantages = (rewards - mean_reward) / std_reward # Shape: [Group_size]
    
    # 广播到 Token 级别 Shape: [Group_size, 1]
    advantages = advantages.unsqueeze(1) 
    
    # 步骤 B: 计算概率比值
    ratio = torch.exp(new_logprobs - old_logprobs) # Shape: [Group_size, Sequence_length]
    
    # 步骤 C: Clip-Higher (解耦裁剪)
    clip_ratio = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high)
    
    # 步骤 D: 计算 Surrogate Loss
    surr1 = ratio * advantages
    surr2 = clip_ratio * advantages
    token_loss = -torch.min(surr1, surr2) 
    
    # 核心修改：使用 Mask 过滤掉 Prompt 和 Padding 部分的 Loss，仅对有效的 Completion Token 求平均
    valid_loss = (token_loss * mask).sum() / mask.sum()
    return valid_loss


def main():
    # 取出问题 1 及其组内回答、得分
    prompt = dapo_dataset["prompt"][0]
    completions = dapo_dataset["completions_group"][0]
    rewards = dapo_dataset["rewards_group"][0]

    # 动态采样 (Dynamic Sampling)
    if rewards.std() == 0:
        print("触发动态采样：本组样本准确率全为 1 或全为 0，跳过训练！")
    else:
        # 1. 将文本转化为对齐的 Tensor 和 Mask
        input_ids, mask = prepare_group_batch(prompt, completions, tokenizer)
        
        # 2. 获取旧策略的对数概率 (模拟 Reference Model 或者旧的 Checkpoint)
        # 实际训练中，这些可能是你在 roll-out 阶段顺手存下来的无梯度数据
        with torch.no_grad():
            old_logprobs = get_logprobs(model, input_ids)
            
        # 3. 真实模型前向传播 (开启梯度图)
        new_logprobs = get_logprobs(model, input_ids)
        # print(new_logprobs.shape)  # 应该是 [Group_size, Sequence_length] torch.Size([4, 23])
        # 4. 计算带有 Mask 保护的 DAPO Loss
        loss = compute_dapo_loss(old_logprobs, new_logprobs, rewards, mask)
        
        print(f"--- 真实 DAPO 模型训练步骤 ---")
        print(f"当前输入 Batch 形状 (Group_size, Max_seq_len): {input_ids.shape}")
        print(f"本组规则得分: {rewards.tolist()}")
        print(f"真实计算出的 DAPO Loss: {loss.item():.4f}")
        
        # 5. 反向传播更新真实模型参数
        loss.backward()
        print("反向传播完成，梯度已记录！(可执行 optimizer.step() 更新权重)")

if __name__ == "__main__":
    main()