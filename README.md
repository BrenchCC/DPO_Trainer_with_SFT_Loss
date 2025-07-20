# DPO Trainer with SFT Loss
一个基于 Direct Preference Optimization (DPO) 训练的模块化框架，支持结合SFT损失以及Lora微调的训练。



## 监控指标

本模块会记录以下指标：
- `dpo_loss`: DPO损失值
- `dpo_accuracy`: DPO准确率（chosen > rejected的比例）
- `sft_loss`: SFT损失值（如果启用）
- `policy_diff_mean`: 策略模型对chosen和rejected的概率差异
- `kl_chosen_mean`/`kl_rejected_mean`: KL散度统计
