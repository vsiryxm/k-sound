# k-sound 项目交接提示词（给下一个 AI Agent）

> 把本文件全部内容作为新会话的第一条消息发给下一个 agent 即可。

---

## 你的身份与任务
你是接手「k-sound」项目的全栈开发工程师。项目目标：本地 AI 音色翻唱 DEMO——用户上传一段本人干声 + 一首歌，生成「用本人音色唱这首歌」的成品音频，可在线试听、下载。本机 localhost 运行。

## 当前状态（重要）
- M1（后端模型跑通）✅ 已完成
- 完整歌曲端到端验证 ✅ 已完成，用户四维初评通过（音色像本人 / 吐词清晰 / 没跑调 / 无严重伪影）
- M2（前端联调）❌ 尚未开始 ← **这是你的主要任务**

## 你的任务：完成 M2
把已验证的流程封装成网页工具（本机 localhost）：
1. **FastAPI 后端**，任务式接口：
   - `POST /jobs`  上传 voice + song → 返回 job_id
   - `GET /jobs/{id}`  返回阶段/状态/错误
   - `GET /jobs/{id}/result`  下载成品
2. **前端**：单页原生 HTML + JS（无框架）：
   - 上传两个文件 → 创建任务 → 轮询 `GET /jobs/{id}` 更新进度 → 试听/下载
   - 授权勾选（本人音色授权）
3. 完成后可直接交付用户使用。

## 项目环境（必须沿用，勿改）
- 项目根目录：`/Users/xinmin/workspace/ai-project/k-sound`
- Python：用项目自带 venv `.venv/`（Python 3.11.16）。系统 `python3` 是 3.14，不兼容，**绝对不要用**。
- 包管理：uv 在 `/Users/xinmin/.local/bin/uv`
- ffmpeg：`/opt/homebrew/bin/ffmpeg`（9.0.2）
- 机器：macOS M5 Max / arm64 / 64GB / torch 2.4.0 MPS

## 已验证的推理脚本
- `engine/infer_svc.py` —— SEED-VC 换音色，已跑通。
  - CLI：`--source <人声wav> --target <干声> --output <wav> --diffusion-steps 30`
  - 已内置：HF 镜像、MPS 设备选择、autocast 修复、分块扩散 + BigVGAN + crossfade、RMVPE 分块 F0

## 完整链路（M2 后端要复刻这三步）
1. **分轨（Demucs）**：
   ```
   HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 .venv/bin/python -m demucs.separate -n htdemucs --two-stems vocals -o <out_dir> <song>
   ```
   → 产出 `vocals.wav` / `no_vocals.wav`（44.1kHz 16-bit 立体声）
2. **换音色（SEED-VC）**：
   ```
   .venv/bin/python engine/infer_svc.py --source <vocals.wav> --target <voice> --output <converted.wav> --diffusion-steps 30
   ```
   → 产出 `converted.wav`（**float32 单声道**！）
3. **回混**：
   用 `soundfile` 读 converted.wav(单声道) + no_vocals.wav(立体声)；人声复制成双声道；人声增益 1.0 / 伴奏 0.9；相加峰值 >1 则整体缩放到 0.95；写 44.1kHz 16-bit wav，再 ffmpeg 转 192k mp3。成品文件名加 `_AI` 后缀。

## 四个关键坑（重蹈会翻车，务必沿用）
1. HuggingFace 下载必须带 `HF_ENDPOINT=https://hf-mirror.com` + `HF_HUB_DISABLE_XET=1`（否则超时 / 卡 0 字节）。
2. torch 2.4.0 的 MPS **不支持** `torch.autocast(device_type='mps')`，只在 CUDA 用 fp16，MPS/CPU 直接 float32。
3. SEED-VC 对 >30s 长音频提 F0 会 MPS 崩溃（`Failed to allocate IOGPUDeviceShmem`），已用分块修复（`infer_svc.py` 的 `rmvpe_infer_chunked`）。
4. SEED-VC 输出 converted.wav 是 float32 单声道；**混音时不要再除以 32768**（曾因此把人声压成静音）。

## 模型 / checkpoint
- singing checkpoint：`checkpoints/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth`（44.1kHz f0-condition；**勿用 22.05kHz 语音 VC**）
- config：`config_dit_mel_seed_uvit_whisper_base_f0_44k.yml`（路径见 `engine/infer_svc.py` 内的配置变量）
- campplus：`engine/seed-vc/campplus_cn_common.bin`
- whisper-small / bigvgan / rmvpe / HTDemucs：已在 `~/.cache/huggingface/hub/`

## 验收标准（PRD 精简）
- 功能：音色上传、歌曲上传、翻唱生成、分步进度、试听下载、授权勾选
- 性能：3–5 分钟歌全流程 ≤5 分钟（实测约 3.7 分钟：分轨 11s + 换音色 208s + 回混 2s）
- 质量四维：相似度 ≥3.5 / 歌词可懂 / 不跑调 / 无严重伪影（已初步通过）

## 合规
- 成品加 `_AI` 后缀
- 仅本人音色授权 ≠ 歌曲版权，商用需另授权
- 不下载盗版 / 版权歌（源歌由用户自己提供本地文件）

## 参考文档（先读这些再动手）
- `docs/PRD.md`（完整需求）
- `docs/M1-report.md`（M1 验证指标）
- `engine/infer_svc.py`（核心推理，直接复用其逻辑）

## 交付要求
M2 完成后：用户能在本机浏览器打开 localhost，上传干声 + 歌 → 点开始 → 看到分步进度 → 最后试听 + 下载成品。

> AI生成
