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
            "Python",                        # 奖励: 1.0 (短，不应受惩罚)
            "Python is great for AI.",       # 奖励: 1.0 (中长，可能会受软惩罚)
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
# 辅助函数 1：拼接、补齐(Padding)并生成 Mask
# ==========================================
def prepare_group_batch(prompt, completions, tokenizer):
    prompt_ids = tokenizer('USER:' + prompt + 'ASSISTANT:', add_special_tokens=True)['input_ids']
    
    input_ids_list = []
    mask_list = []
    
    for comp in completions:
        comp_ids = tokenizer(comp, add_special_tokens=False)['input_ids']
        input_ids_list.append(prompt_ids + comp_ids)
        mask_list.append([0] * len(prompt_ids) + [1] * len(comp_ids))
        
    max_length = max(len(ids) for ids in input_ids_list)
    
    padded_input_ids = []
    padded_masks = []
    
    for ids, mask in zip(input_ids_list, mask_list):
        pad_len = max_length - len(ids)
        padded_input_ids.append(ids + [pad_id] * pad_len)
        padded_masks.append(mask + [0] * pad_len)
        
    return torch.tensor(padded_input_ids), torch.tensor(padded_masks)

# ==========================================
# 辅助函数 2：提取 Token 的 Log 概率
# ==========================================
def get_logprobs(model, input_ids):
    logits = model(input_ids).logits
    log_probs = logits.log_softmax(dim=-1)
    per_token_logps = torch.gather(log_probs, dim=2, index=input_ids.unsqueeze(2)).squeeze(2)
    return per_token_logps

# ==========================================
# 辅助函数 3：DAPO 软过长惩罚 (Soft Overlong Punishment)
# ==========================================
def apply_soft_length_penalty(base_rewards, completion_lengths, start_len, max_len, max_penalty=0.5):
    """
    根据回答长度，对超长回答进行动态惩罚，减少奖励噪声
    """
    # 1. 计算超出起罚点的长度
    excess_len = (completion_lengths - start_len).clamp(min=0.0)
    
    # 2. 计算惩罚比例 (0 到 1 之间)
    penalty_ratio = (excess_len / (max_len - start_len)).clamp(max=1.0)
    
    # 3. 计算最终扣分
    penalties = penalty_ratio * max_penalty
    
    # 4. 基础分减去惩罚分
    shaped_rewards = base_rewards - penalties
    return shaped_rewards

# ==========================================
# 2. DAPO 核心算法: 计算 Token-Level Loss 与 Clip-Higher
# ==========================================
def compute_dapo_loss(old_logprobs, new_logprobs, rewards, mask, eps_low=0.2, eps_high=0.28):
    # 步骤 A: 计算组内相对优势 (Advantage Normalization)
    mean_reward = rewards.mean()
    std_reward = rewards.std() + 1e-8
    advantages = (rewards - mean_reward) / std_reward # Shape: [Group_size]
    advantages = advantages.unsqueeze(1) # 广播到 Token 级别 Shape: [Group_size, 1]
    
    # 步骤 B: 计算概率比值
    ratio = torch.exp(new_logprobs - old_logprobs) # Shape: [Group_size, Sequence_length]
    
    # 步骤 C: Clip-Higher (解耦裁剪) - 突破 PPO 的对称性
    clip_ratio = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high)
    
    # 步骤 D: 计算 Surrogate Loss
    surr1 = ratio * advantages
    surr2 = clip_ratio * advantages
    token_loss = -torch.min(surr1, surr2) 
    
    # 使用 Mask 过滤 Prompt 和 Padding，仅计算有效回答 Token 的平均 Loss
    valid_loss = (token_loss * mask).sum() / mask.sum()
    return valid_loss


def main():
    prompt = dapo_dataset["prompt"][0]
    completions = dapo_dataset["completions_group"][0]
    base_rewards = dapo_dataset["rewards_group"][0]

    # 动态采样过滤
    if base_rewards.std() == 0:
        print("触发动态采样：本组样本准确率全为 1 或全为 0，跳过训练！")
    else:
        # 1. 将文本转化为对齐的 Tensor 和 Mask
        input_ids, mask = prepare_group_batch(prompt, completions, tokenizer)
        
        # ----------------------------------------------------
        # [新增模块]: DAPO 过长奖励整形
        # ----------------------------------------------------
        # 统计每个回答的真实 Token 数量 (Mask 中为 1 的数量就是回答的长度)
        completion_lengths = mask.sum(dim=1).float()
        
        # 为了演示，我们将起罚长度设得非常小 (2 个 Token)，最大惩罚设为 0.5
        # 实际训练中，这些值通常很大 (例如 start_len=16384, max_len=20480)
        START_LEN = 2.0
        MAX_LEN = 5.0
        MAX_PENALTY = 0.5
        
        shaped_rewards = apply_soft_length_penalty(
            base_rewards, 
            completion_lengths, 
            START_LEN, 
            MAX_LEN, 
            MAX_PENALTY
        )
        # ----------------------------------------------------

        # 2. 获取旧策略的对数概率
        with torch.no_grad():
            old_logprobs = get_logprobs(model, input_ids)
            
        # 3. 真实模型前向传播
        new_logprobs = get_logprobs(model, input_ids)
        
        # 4. 计算 DAPO Loss (注意：这里传入的是整形后的 shaped_rewards)
        loss = compute_dapo_loss(old_logprobs, new_logprobs, shaped_rewards, mask)
        
        print(f"--- 真实 DAPO 模型训练步骤 ---")
        print(f"当前输入 Batch 形状: {input_ids.shape}")
        print(f"各回答真实长度 (Tokens): {completion_lengths.tolist()}")
        print(f"基础规则得分 (Base Rewards): {base_rewards.tolist()}")
        print(f"软惩罚后得分 (Shaped Rewards): {[round(x, 2) for x in shaped_rewards.tolist()]}")
        print(f"\n真实计算出的 DAPO Loss: {loss.item():.4f}")
        
        # 5. 反向传播更新真实模型参数
        loss.backward()
        print("反向传播完成，梯度已记录！(可执行 optimizer.step() 更新权重)")

if __name__ == "__main__":
    main()