# Podcast Transcription

一个面向长中文播客和访谈的 Codex skill，配套本地 Qwen3-ASR 转写流水线。

把本仓库作为 skill 安装到 Codex skills 目录后，Codex 会在用户提供本地音频并要求转写、时间戳、说话人区分或摘要时使用它。

## 本地流水线

`pipeline/` 提供可续跑的本地处理脚本。模型权重不会提交到仓库，需要首次运行 `pipeline/setup.sh` 下载。

```bash
cd pipeline
./run-cpu-sample.sh
./run-gpu-sample.sh
./run-gpu.sh
```

GPU 流程按 RTX 2070 Super 8GB 配置：FP16、SDPA、batch size 1；ASR 和时间戳对齐分开运行。结果写到 `pipeline/outputs/`，其中原始识别稿和对齐结果分开保存。

## 目录

- `SKILL.md`：Codex 使用说明
- `agents/openai.yaml`：界面元数据
- `references/local-pipeline.md`：本地流水线说明
- `pipeline/`：环境、转写、对齐和检查脚本

该仓库只包含代码和说明，不包含播客音频、模型权重或转写内容。
