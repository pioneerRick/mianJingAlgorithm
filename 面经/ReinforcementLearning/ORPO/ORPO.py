import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer

# ==========================================
# 0. 初始化真实的模型与分词器
# ==========================================
config = LlamaConfig(vocab_size=32000, hidden_size=256, num_hidden_layers=2)
model = LlamaForCausalLM(config) # 同样，不需要 ref_model！

tokenizer = AutoTokenizer.from_pretrained('HuggingFaceM4/tiny-random-LlamaForCausalLM')
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id or 0
pad_id = tokenizer.pad_token_id

# ==========================================
# 1. 准备 ORPO 数据集 (基于 KTO 数据改造为成对数据)
# ==========================================
orpo_dataset = {
    "prompt": [
        "How are you", 
        "What is your name?", 
        "Which is the best programming language?",
        "How to kill a man?"
    ],
    "chosen": [
        "hi nice to meet you",                  # 好回答
        "I am an AI assistant.",                # 补充的好回答
        "Python",                               # 好回答
        "I cannot fulfill this request."        # 补充的安全回答
    ],
    "rejected": [
        "I don't care.",                        # 补充的坏回答
        "leave me alone",                       # 坏回答
        "C++",                                  # 补充的被拒绝回答
        "Use gun shoot his head"                # 坏回答
    ]
}

# ==========================================
# 辅助函数：拼接、补齐(Padding)并生成 Mask
# ==========================================
def prepare_batch(prompts, completions, tokenizer):
    input_ids_list = []
    mask_list = []
    
    for prompt, comp in zip(prompts, completions):
        prompt_ids = tokenizer('USER:' + prompt + 'ASSISTANT:', add_special_tokens=True)['input_ids']
        comp_ids = tokenizer(comp, add_special_tokens=False)['input_ids']
        
        input_ids_list.append(prompt_ids + comp_ids)
        # Mask: Prompt 部分为 0，回答部分为 1
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
# 2. ORPO 核心算法 (完全对齐图 5 和图 6 的逻辑)
# ==========================================
def compute_orpo_loss(model, chosen_inputs, chosen_masks, rejected_inputs, rejected_masks, lambda_weight=0.1):
    """
    计算 ORPO 损失: L_ORPO = L_SFT + lambda * L_OR
    """
    # ---------------------------------------------------------
    # 步骤 A: 前向传播，获取 Logits
    # ---------------------------------------------------------
    chosen_logits = model(chosen_inputs).logits
    rejected_logits = model(rejected_inputs).logits

    # ---------------------------------------------------------
    # 步骤 B: 计算 L_SFT (仅在 chosen 样本上计算标准的监督微调损失)
    # 取 logits 的前 N-1 个去预测 input_ids 的后 N-1 个
    # ---------------------------------------------------------
    shift_logits = chosen_logits[..., :-1, :].contiguous()
    shift_labels = chosen_inputs[..., 1:].contiguous()
    shift_mask = chosen_masks[..., 1:].contiguous()
    
    # 手动计算交叉熵以应用 Mask (忽略 Prompt 和 Padding)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    sft_loss_per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    sft_loss_per_token = sft_loss_per_token.view(shift_labels.size())
    
    # 得到只针对好回答的 SFT Loss
    sft_loss = (sft_loss_per_token * shift_mask).sum() / shift_mask.sum()

    # ---------------------------------------------------------
    # 步骤 C: 提取每个回答的平均对数似然 (对应图 5 代码)
    # log P_theta (y|x)
    # ---------------------------------------------------------
    def get_avg_logps(logits, input_ids, mask):
        log_probs = logits.log_softmax(dim=-1)
        per_token_logps = torch.gather(log_probs, dim=2, index=input_ids.unsqueeze(2)).squeeze(2)
        # 对应图5：长度归一化 (除以 mask 的和，消除长度偏见)
        avg_logps = (per_token_logps * mask).sum(dim=-1) / mask.sum(dim=-1)
        return avg_logps

    chosen_avg_logps = get_avg_logps(chosen_logits, chosen_inputs, chosen_masks)
    rejected_avg_logps = get_avg_logps(rejected_logits, rejected_inputs, rejected_masks)

    # ---------------------------------------------------------
    # 步骤 D: 计算胜算比损失 L_OR (对应图 6 代码)
    # log_odds = log(P / (1-P)) = log(P) - log(1-P)
    # 在数值计算中: log(1-P) = torch.log1p(-torch.exp(log_P))
    # ---------------------------------------------------------
    log_odds_chosen = chosen_avg_logps - torch.log1p(-torch.exp(chosen_avg_logps))
    log_odds_rejected = rejected_avg_logps - torch.log1p(-torch.exp(rejected_avg_logps))
    
    # log (odds_chosen / odds_rejected)
    log_odds_diff = log_odds_chosen - log_odds_rejected
    
    # L_OR = -log(sigmoid(log_odds_diff))
    l_or = -F.logsigmoid(log_odds_diff).mean()

    # ---------------------------------------------------------
    # 步骤 E: 合并最终目标函数 (对应图 3 公式)
    # ---------------------------------------------------------
    total_loss = sft_loss + lambda_weight * l_or
    
    return total_loss, sft_loss, l_or


def main():
    # 处理选中和拒绝的数据，变为模型可接受的 Tensor
    chosen_inputs, chosen_masks = prepare_batch(orpo_dataset["prompt"], orpo_dataset["chosen"], tokenizer)
    rejected_inputs, rejected_masks = prepare_batch(orpo_dataset["prompt"], orpo_dataset["rejected"], tokenizer)
    
    # 超参数 lambda (图 3 公式 4 中的 λ，图 6 代码中的 self.beta)
    # 用于调节偏好对齐强度和微调效果之间的平衡
    LAMBDA_WEIGHT = 0.1 

    # 执行 ORPO 前向与 Loss 计算
    total_loss, sft_loss, l_or = compute_orpo_loss(
        model, 
        chosen_inputs, chosen_masks, 
        rejected_inputs, rejected_masks, 
        lambda_weight=LAMBDA_WEIGHT
    )
    
    print(f"--- ORPO 算法运行结果 ---")
    print(f"1. 监督微调损失 (L_SFT): {sft_loss.item():.4f} (负责让模型学会说人话)")
    print(f"2. 胜算比损失 (L_OR): {l_or.item():.4f} (负责拉开好坏回答的概率差距)")
    print(f"3. 最终总损失 (Total Loss): {total_loss.item():.4f}")
    
    # 反向传播更新模型
    total_loss.backward()
    print("反向传播完成，模型已通过单阶段融合更新！")

if __name__ == "__main__":
    main()