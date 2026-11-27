# k-sound 安装代执行提示词（交给 AI 编程助手使用）

> 用法：克隆本仓库后，把本文档**全文**发给任意 AI 编程助手（Claude Code / Cursor / Codex / 其他终端 agent），并附一句「请按这份文档帮我安装并验收」。文档即完整的执行指令：按顺序做、逐项核对、不过验收不结束。

---

## 你的任务

在当前机器上把 k-sound（本地 AI 音色翻唱 Web 服务）安装到可运行状态，并通过验收清单。全程不要修改仓库代码；只在明确允许的目录里创建文件。

**硬性边界**：

- 不要动 `engine/seed-vc/` 内部任何文件（第三方源码，锁定 commit）
- 不要提交或推送 git
- 不要安装文档清单之外的包
- 下载总计约 4GB（依赖 2GB + 模型 2GB），开始前确认磁盘余量 ≥ 10GB

## 第 0 步 · 前置环境检查（不满足先报告，不要硬装）

逐项检查并输出结果表：

| 检查项 | 命令 | 通过标准 |
|---|---|---|
| 操作系统 | `uname -sm` | `Darwin arm64`（Apple Silicon Mac；Linux x86_64 + NVIDIA 有回退路径但未验证，遇此情况向用户说明风险后继续） |
| 内存 | `sysctl hw.memsize` | ≥ 16GB（低于则提示体验受限，经用户确认后继续） |
| 磁盘余量 | `df -h .` | ≥ 10GB |
| ffmpeg | `ffmpeg -version` | 任意版本可用；没有则 `brew install ffmpeg`（Linux 用 apt 等价命令） |
| git | `git --version` | 可用 |

## 第 1 步 · 获取 SEED-VC 源码（仓库未内置，必须先做）

```bash
git clone --depth 1 https://github.com/Plachtaa/seed-vc.git engine/seed-vc
cd engine/seed-vc && git checkout 51383ef && cd ../..
```

验收：`ls engine/seed-vc/modules/commons.py` 存在。
注：checkout 到固定 commit 是因为项目按该版本（51383ef）调试；上游更新可能不兼容。clone 失败时依次尝试：① `git config --global http.https.github.com.proxy` 检查并临时解除代理干扰；② 提示用户网络问题，勿静默换源。

## 第 2 步 · Python 3.11 虚拟环境

**禁止**用系统默认 Python（macOS 新版自带 3.13/3.14，torch 2.4 无对应轮子，必装失败）。

优先 uv（能自动下载 3.11，最省事）：

```bash
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh   # 询问用户是否同意安装 uv
uv venv --python 3.11 .venv
```

无 uv 时回退（需用户已装 python3.11）：

```bash
python3.11 -m venv .venv
```

验收：`.venv/bin/python --version` 输出 `Python 3.11.*`。

## 第 3 步 · 安装依赖

```bash
uv pip install -r requirements.txt --python .venv/bin/python
# 或无 uv：.venv/bin/pip install -r requirements.txt
```

- 清单已锁定为实测版本（torch 2.4.0 / demucs 4.1.0 / transformers 4.46.3 等），**不要升级任何包**
- Linux + NVIDIA 显卡：先按 pytorch.org 官方指引换装 cu121 的 torch/torchaudio（版本仍要 2.4.0），再装其余
- macOS arm64 直接从 PyPI 装即自带 MPS 支持，不需要 nightly/cu 索引

验收：`.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"` 输出 `True`（NVIDIA 机器检查 `torch.cuda.is_available()`）。

## 第 4 步 · 启动服务

```bash
./start.sh
```

`start.sh` 会以后台方式拉起 uvicorn 并尝试打开浏览器。若你自己的 agent 环境不方便跑前台脚本，等价命令：

```bash
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8300
```

首次启动**不会**下载模型（模型在第一次生成任务时才按需下载，见第 6 步）。

## 第 5 步 · 验收清单（全部通过才算完成）

| # | 检查 | 命令 | 通过标准 |
|---|---|---|---|
| 1 | 首页 | `curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8300/` | `200` |
| 2 | 设置接口 | `curl -s http://127.0.0.1:8300/api/config` | JSON 含 `config.quality` 与 `qualities` 档位表 |
| 3 | ffmpeg 探测 | `curl -s http://127.0.0.1:8300/api/ping` | `{"ok":true,"ffmpeg":true}` |
| 4 | 历史接口 | `curl -s "http://127.0.0.1:8300/history?limit=1"` | JSON 含 `total` 字段（新装为 0） |
| 5 | 浏览器 | 打开 http://127.0.0.1:8300 | 页面标题「k-sound · 用我的声音唱歌」，音色授权弹窗可见 |

验收 1–4 通过后，向用户报告：服务已就绪，首个翻唱批次将自动下载约 2GB 模型（SEED-VC 783MB + RMVPE 173MB 存 `./checkpoints`；whisper-small 922MB + HTDemucs 80MB 存 `~/.cache/huggingface`），之后运行全程离线。

## 第 6 步 · 端到端冒烟（可选，需要用户提供音频）

如果用户提供了干声和歌曲文件，可代为提交一次任务验收全链路（否则跳过，留给用户在网页上自己操作）：

```bash
curl -s -X POST http://127.0.0.1:8300/jobs \
  -F "voice=@<干声文件路径>" -F "song=@<歌曲文件路径>" \
  -F "quality=fast" -F "authorized=true"
# 用返回的 job_id 轮询：curl -s http://127.0.0.1:8300/jobs/<job_id>
# 标准耗时：快速档约为歌曲时长 ÷ 1.5（首次加 2GB 模型下载时间）
```

首个任务即触发模型下载，观察服务日志出现 huggingface 下载进度且最终 `status: done`、`jobs/<job_id>/` 下生成 `*_AI.mp3` 为通过。

## 已知坑速查（排障时按此优先）

1. **torch 装不上/装错 python**：九成是 Python 版本 > 3.12。确认 `.venv` 是 3.11（`uv venv --python 3.11`）。
2. **模型下载卡住**：代码默认走 `https://hf-mirror.com`（国内镜像）。海外网络若反常，`export HF_ENDPOINT=https://huggingface.co` 后重启服务。已下载的部分有缓存，重试不浪费。
3. **第一次生成比标称慢很多**：正常，含模型下载（约 2GB）。第二个批次起恢复 3–6 分钟/首。
4. **生成中途报 IOGPUDeviceShmem / Failed to allocate**：内存不足。让用户关掉大型应用，改用「快速」档或短歌重试。
5. **服务起不来、端口占用**：`lsof -nP -iTCP:8300 -sTCP:LISTEN` 查占用进程；本项目服务用 TERM 停止后重启即可。
6. **seed-vc 子进程报找不到模块**：检查第 1 步的 commit 是否为 51383ef，且 `engine/seed-vc/modules/` 目录完整。
7. **不要以 root/sudo 安装**；一切装在项目内 `.venv`，不污染系统环境。

## 完成报告格式

安装结束后向用户输出：

```
✅/❌ 环境检查：…（逐项）
✅/❌ SEED-VC：commit 51383ef，路径 engine/seed-vc/
✅/❌ Python：.venv 3.11.x
✅/❌ 依赖：requirements.txt 全量安装，MPS/CUDA 可用
✅/❌ 服务：http://127.0.0.1:8300 验收 1–5 通过
➡️ 下一步：用户上传干声+歌曲生成第一个翻唱（首跑自动下载约 2GB 模型）
```

任何一步无法满足时，停在该步，把「检查结果 + 尝试过的命令 + 错误原文」报告给用户，不要跳过或伪造通过。

> AI生成
