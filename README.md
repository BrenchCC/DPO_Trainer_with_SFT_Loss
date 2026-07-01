# DPO Trainer with SFT Loss

一个基于 Hugging Face Trainer 的轻量训练项目，支持标准 SFT、DPO，以及在 DPO 目标上叠加 chosen-response SFT loss。

## 功能

- 将多轮 `messages` 中的每个 assistant 回复展开为独立 SFT 样本
- 对 prompt token 使用 `-100` mask，仅监督 assistant 回复
- DPO chosen/rejected 共享完全相同的 prompt token
- 支持 DPO + 加权 SFT loss
- 支持 LoRA；在 CUDA/Linux 环境中可选支持 QLoRA
- 支持动态 padding、验证集和训练配置保存

## 安装与验证

项目要求 Python 3.10 或更高版本。

```bash
pip install -r requirements.txt
python validate_setup.py
python -m unittest discover -s tests -v
```

QLoRA 需要 CUDA、兼容的 NVIDIA GPU，并额外安装：

```bash
pip install bitsandbytes
```

## 数据格式

### SFT

每条记录包含一个非空 `messages` 列表。支持 `system`、`user` 和 `assistant` 角色；每个 assistant 轮次都会生成一个训练样本。

```json
[
  {
    "messages": [
      {"role": "system", "content": "你是一个有帮助的助手。"},
      {"role": "user", "content": "解释 DPO。"},
      {"role": "assistant", "content": "DPO 是一种偏好优化方法。"},
      {"role": "user", "content": "它需要奖励模型吗？"},
      {"role": "assistant", "content": "标准 DPO 不需要单独训练奖励模型。"}
    ]
  }
]
```

### DPO

`instruction`、`chosen` 和 `rejected` 必须为非空字符串。`input` 和 `history` 可选。

```json
[
  {
    "instruction": "解释 DPO。",
    "input": "请使用简洁语言。",
    "chosen": "DPO 直接使用偏好数据优化语言模型。",
    "rejected": "DPO 就是普通监督学习。",
    "history": [["什么是偏好数据？", "偏好数据包含更好和较差的回答。"]]
  }
]
```

`history` 也可以使用标准消息格式：

```json
"history": [
  {"role": "user", "content": "什么是偏好数据？"},
  {"role": "assistant", "content": "偏好数据包含更好和较差的回答。"}
]
```

## 训练

### 标准 SFT

```bash
python train.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --train_data_path data_examples/sft_data_example.json \
  --output_dir output_sft \
  --dpo false \
  --lora true \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --report_to none
```

### DPO + SFT Loss

```bash
python train.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --train_data_path data_examples/dpo_data_example.json \
  --output_dir output_dpo \
  --dpo true \
  --lora true \
  --dpo_beta 0.1 \
  --sft_loss_weight 0.1 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --remove_unused_columns false \
  --report_to none
```

也可以直接运行 `bash scripts/train_sft.sh` 或 `bash scripts/train_dpo.sh`。脚本支持通过 `MODEL_PATH`、`TRAIN_DATA`、`EVAL_DATA`、`OUTPUT_DIR` 和 `LORA` 环境变量覆盖默认配置。

本项目以 `--dpo`、`--lora` 为主参数，同时兼容参考仓库的 `--use_dpo`、`--use_lora`。

## 截断和损失范围

- response 过长时从右侧截断并保留 EOS
- prompt 过长时从左侧截断，优先保留最近上下文和回复
- DPO 对中的 chosen/rejected 使用同一个 prompt 截断结果
- SFT labels 只覆盖 assistant response 和 EOS；prompt 与 padding 均为 `-100`

## DPO 指标

- `dpo_loss`：DPO 目标损失
- `dpo_accuracy`：chosen reward 高于 rejected reward 的比例
- `sft_loss`：启用混合目标时的 chosen-response SFT 损失
- `policy_diff_mean`：策略模型 chosen/rejected log probability 差
- `reference_diff_mean`：参考模型 chosen/rejected log probability 差
- `kl_chosen_mean`、`kl_rejected_mean`：策略与参考模型的 log probability 差

训练完成后，模型、tokenizer 和 `training_config.json` 会保存到 `output_dir`。
