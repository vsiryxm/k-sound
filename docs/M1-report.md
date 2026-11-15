# k-sound · M1 验证报告（SEED-VC singing 转换跑通）

日期：2026-11-15
状态：**M1 核心目标达成** —— SEED-VC singing checkpoint（f0-condition，44.1kHz）单次转换成功，产出可试听结果。

## 1. 结论速览

| 项 | 结果 |
|---|---|
| 引擎 | SEED-VC zero-shot 歌声转换，`seed-uvit-whisper-base`（44.1kHz，BigVGAN） |
| 设备 | Apple Silicon **MPS 原生运行**，全程无 CPU fallback、无报错 |
| 输出 | 28.18s 完整可试听音频（源：See You Again 人声 28s → 参考音色 s1p1） |
| 音质指标 | 44.1kHz 单声道，peak 1.0，RMS 0.24，静音占比 12%（正常歌曲停顿） |
| 实时率 | 扩散+声码器阶段 **RTF ≈ 0.836**（快于实时） |

## 2. 环境与依赖

- Python：`3.11.16`（uv 独立 venv，位于项目 `.venv/`；系统默认 3.14 不兼容）
- torch / torchaudio：`2.4.0`（Apple Silicon MPS 版）
- 关键依赖：seed-vc、transformers 4.46.3、librosa、demucs 4.1.0、fastapi、uvicorn
- 系统 ffmpeg：9.0.2

## 3. 遇到并解决的三个问题

1. **HuggingFace 直连超时**：`huggingface.co` 在本机不可达。改用 `HF_ENDPOINT=https://hf-mirror.com`。
2. **xet 存储文件下载卡死（关键）**：hf-mirror 对 xet 存储的文件会 302 到 `cas-bridge.xethub.hf.co`，导致 whisper-small / bigvgan / rmvpe 下载卡在 0 字节。修复：`HF_HUB_DISABLE_XET=1`，改走镜像 LFS 代理，下载速度约 3MB/s。Seed-VC 的 `.pth` 是旧式 LFS，本就正常。
3. **torch 2.4.0 的 MPS 不支持 `torch.autocast`**：官方 SEED-VC 代码用 `torch.autocast(device_type='mps')`，在 torch 2.4 直接抛 `unsupported autocast device_type 'mps'`。修复：MPS/CPU 跳过 autocast，直接 float32 推理（CUDA 才用 fp16 autocast）。

## 4. 分阶段耗时（28s 源音频实测）

| 阶段 | 耗时 |
|---|---|
| 模型加载（全部权重已缓存） | 10.98s |
| 加载音频 + 重采样 44.1kHz | 0.21s |
| Whisper 语义特征提取 | 0.29s |
| RMVPE F0 提取 + 音高对齐 | 4.42s |
| Length regulator | 0.20s |
| 扩散解码 + BigVGAN（2 块） | 23.56s |
| **合计（不含模型加载）** | **28.69s** |

## 5. 性能记录（M1 实测）

- 峰值内存（进程 RSS）：**≈ 2282 MB**
- MPS 当前分配 / 驱动分配：1881 MB / 5348 MB
- 是否 MPS fallback：**否**（未启用 fallback，无 unsupported op 报错）
- 分块接缝：28s 源音频拆为 **2 块**（16 帧重叠 crossfade），需长音频实测确认接缝听感
- **3 分钟歌曲估算**（线性外推，未实测）：F0 ≈ 28s + 扩散 ≈ 150s + 其余 ≈ 3s ≈ **3 分钟**，符合「≤5 分钟」初始目标；待真实 3 分钟歌曲实测校准

## 6. 尚未验证（M1 遗留，进入 M2/M3 前补齐）

1. **真实音色**：本次参考音色用的是仓库自带 `s1p1.wav`（说话样本），非本人音色。音色相似度四维初评需本人真实干声。
2. **完整歌曲 + 分轨混音**：本次直接用了仓库预分离的 `[vocals]` 人声，未跑 Demucs 分离与伴奏回混。Demucs（htdemucs）依赖已装、模型已缓存，待真实歌曲验证。
3. **3 分钟长歌**：接缝与总耗时的正式校准需要一首 3–5 分钟完整歌曲。

## 7. 下一步

等本人提供：① 10–30 秒清晰干声（本人音色）② 一首 mp3/wav/m4a 完整歌曲，即可跑「Demucs 分轨 → SEED-VC 换音色 → 伴奏回混」完整链路，并由本人按四维度（音色相似度 / 歌词可懂 / 音高节奏 / 伪影）正式打分。
