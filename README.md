# k-sound · 用我的声音唱歌

上传一段**你的干声**和一首**目标歌曲**，本地生成用你的音色演唱的翻唱成品。全程本机推理，音频不上传、不联网、无账号。

> 本人音色授权 ≠ 歌曲词曲/录音版权。生成的成品文件名自动追加 `_AI` 后缀，仅供个人本地使用，商用需自行取得授权。

## 它能做什么

- **零样本音色转换**：10–15 秒干声即可提取音色，无需训练（基于 [SEED-VC](https://github.com/Plachtaa/seed-vc) 歌声模型）
- **全自动流水线**：人声/伴奏分离（[Demucs htdemucs](https://github.com/adefossez/demucs)）→ 音色转换 → 混音导出，一首 4 分半的歌约 3–6 分钟
- **三音源对比试听**：网页上一键切换「翻唱 / 原唱 / 我的干声」，进度保持
- **分轨导出**：AI 人声（ai.wav）与伴奏（bg.wav）单独下载，可导入 DAW 做深度后期
- **批次历史**：SQLite 记录每次生成，含速度/内存/时长统计、质量档、混响参数，批次 ID 可复制
- **傻瓜化设置**：质量三档（扩散 20/30/45 步）、音高 ±12 半音、混响湿度滑条

## 最小运行环境要求

| 项目 | 最低要求 | 说明 |
|---|---|---|
| 机型 | **Apple Silicon Mac**（M1/M2/M3/M4 任一） | 推理走 PyTorch MPS；Linux + NVIDIA 显卡有代码回退路径（CUDA），未充分验证；纯 CPU 可跑但慢到不实用 |
| 系统 | macOS 13+ | MPS 后端要求；ffmpeg 需可用 |
| 内存 | **16GB**（推荐 32GB） | 实测峰值：GPU 驱动占用约 5.4GB + Python 进程约 2.4GB。8GB 机器会频繁换页甚至失败 |
| 磁盘 | **约 8GB**（安装 3GB + 首次下载模型约 2GB + 首批产物 0.4GB） | 模型首次生成时自动下载并缓存；每个翻唱批次占约 380MB，可随时清理 `jobs/` |
| Python | **3.11**（3.10/3.12 大概率可用，未验证；**3.13+ 不可用**，torch 2.4 无预编译轮子） | |
| 其他 | ffmpeg（`brew install ffmpeg`） | 预处理与 MP3 编码依赖 |

**生成速度参考**（4 分 44 秒歌曲，M5 Max 实测）：

| 质量档 | 扩散步数 | 速度 |
|---|---|---|
| 快速 | 20 | ≈1.5× 实时（约 3 分钟） |
| 标准 | 30 | ≈0.8× 实时 |
| 精细 | 45 | ≈0.85× 实时（约 5 分半） |

入门级 M 芯片预计慢 2–4 倍，建议先用「快速」档。

## 快速开始

```bash
# 1. 克隆
git clone <本仓库地址> k-sound && cd k-sound

# 2. 获取 SEED-VC 源码（约 100MB，本仓库未内置，锁定验证版本）
git clone --depth 1 https://github.com/Plachtaa/seed-vc.git engine/seed-vc
cd engine/seed-vc && git checkout 51383ef && cd ../..

# 3. 创建 Python 3.11 虚拟环境（uv 会自动下载 3.11；无 uv 用 python3.11 -m venv .venv）
uv venv --python 3.11 .venv

# 4. 安装依赖（约 2GB，含 torch）
uv pip install -r requirements.txt --python .venv/bin/python

# 5. 装好 ffmpeg 后启动
./start.sh    # 或：.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8300
```

浏览器打开 <http://127.0.0.1:8300>，勾选音色授权后即可使用。

**首次生成会自动下载模型（约 2GB，一次性）**：SEED-VC 歌声模型 783MB、RMVPE 173MB（存至 `./checkpoints`）、whisper-small 922MB、HTDemucs 80MB（存至 `~/.cache/huggingface`）。之后完全离线运行。

国内网络默认走 [hf-mirror](https://hf-mirror.com) 镜像（代码内置）；海外网络可显式指回官方源：

```bash
export HF_ENDPOINT=https://huggingface.co
```

## 使用说明

1. 上传 **10–15 秒单人清唱干声**（系统自动截取前 15 秒做音色参考；更长的完整上传件仅供试听回放）
2. 上传目标歌曲（**录音室版本**效果最好；现场版/合唱/纯音乐不适用）
3. 勾选「音色为本人所有」授权 → 点「生成翻唱」→ 等 3–6 分钟
4. 历史卡片上可切换三音源对比、下载 MP3/WAV、下载 ai/bg 分轨

可选设置：质量档（快速/标准/精细）、音高移调（±12 半音，适配音色音域差异）、混响湿度（0–30%）。

## 常见问题

- **Python 版本装错**：项目必须 Python ≤3.12（推荐 3.11）。macOS 新系统自带 3.14 装不上 torch 2.4。
- **首次生成卡在「下载模型」**：网络问题。国内确认能访问 hf-mirror.com；海外执行 `export HF_ENDPOINT=https://huggingface.co` 后重启服务。已下载的模型不会重复下载。
- **生成时报内存不足**：16GB 机器请关掉大型应用，或用「快速」档、选短一点的歌。
- **生成的文件在哪**：`jobs/<批次ID>/`，成品 MP3 文件名带 `_AI` 后缀。
- **想清理磁盘**：`jobs/` 目录可整目录删除（历史记录会显示「成品文件已清理」），`~/.cache/huggingface` 别删（模型缓存）。

## 让 AI 帮你安装

已准备好一份给 AI 编程助手（Claude Code / Cursor / Codex 等）的安装提示词，克隆仓库后把 `docs/INSTALL-Agent.md` 全文喂给任意 agent，即可代你完成安装与验收。

## 技术架构一览

```
上传干声(+song) → ffmpeg 预处理(干声截15s) → Demucs htdemucs 两轨分离
  → SEED-VC 零样本歌声转换(扩散 20/30/45 步) → 人声+混响 → 回混导出
  → 44.1kHz WAV + 192k MP3(_AI 后缀) + ai/bg 分轨
```

- 后端：FastAPI 任务队列（单 worker）、SQLite 批次库、进度轮询接口
- 前端：单文件 `static/index.html`，无构建依赖
- 质检：方波伪影检测（引擎输出异常只标记不拦截）

更详细的设计与调优记录见 `docs/`（PRD、里程碑报告、端到端验证）。

## 致谢与许可

核心能力来自社区，感谢：

- [SEED-VC](https://github.com/Plachtaa/seed-vc)（Plachtaa）— 零样本音色转换
- [Demucs](https://github.com/adefossez/demucs)（Facebook Research）— 音源分离
- [whisper](https://github.com/openai/whisper)（OpenAI）— 语音特征
- [pedalboard](https://github.com/spotify/pedalboard)（Spotify）— 音频效果链

本项目代码可自由使用与修改；使用时请遵守你所在地区关于声音权与音乐版权的法律规定。**仅对你本人的音色使用本工具**；生成的翻唱作品的权利归属遵循原歌曲的版权授权。

> AI生成
