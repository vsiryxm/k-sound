# 竞品逆向：浪音 AISing v2.5.1.6 — 结论与偷师清单

> 2026-11-15 对 `竞品/AISing.exe` 做的完整逆向分析（安装壳 → Electron 真身 → asar/.jsc 源码 → VST RTTI）。
> 逆向材料在 `.temp/aising-extract/`（不入库）；本文只沉淀结论与决策，供后续迭代对照。

---

## 一、竞品架构结论

| 维度 | 浪音 AISing | k-sound（我方） |
|---|---|---|
| 推理位置 | **纯云端**（阿里 OSS 分片上传 → 云端分离+推理 → 多镜像下载分轨 ZIP） | **纯本地**（Demucs + SEED-VC，MPS） |
| 音色方案 | **训练型三档**（快速/专业/旗舰，40 分钟样本云端队列训练，出专属音色模型） | 零样本（SEED-VC，15 秒参考干声） |
| 交付形态 | bg + ai **分轨** ZIP + 本地四轨播放器（BASS.dll） | 混音成品单文件（本轮已补分轨） |
| 本地 DSP | AISingerVST.dll/.vst3（JUCE 8.0.3：Freeverb + IIR + 压缩 + WASAPI 低延迟 + 命名管道 IPC 实时 ducking） | pedalboard 软件链（HP→压缩→混响→Haas） |
| 质检 | ffmpeg astats 满幅方波检测（1 秒帧 × 连续 2 帧） | 本轮已补（见下） |
| 其他 | 调性检测、去和声、±12 半音、双设备路由（监听/输出分离） | 音高微调 + 自动 F0 已有 |

**竞品"融合感更好"的三个真实来源**（按贡献排序）：
1. 训练型音色模型的贴近度 >> 零样本（根因，短期难追平）
2. **分轨架构**：AI 人声干轨走用户自己的声卡 VST 效果链（专业链路天然融合）
3. 云端成品可能自带空间处理（客户端本身无本地混响 DSP）

## 二、偷师清单（按性价比排序）

### ① 分轨化 ✅ 本轮已落地（2026-11-15）
- stage4 除 mix 外落盘 `jobs/<id>/ai.wav`（处理链后的 AI 人声）与 `bg.wav`（Demucs 伴奏）
- `GET /jobs/{id}/result?fmt=wav&kind=vocal|bg` 分轨下载（WAV 无损，文件名 `_AI人声/_伴奏`）
- `GET /jobs/{id}/audio/{kind}` 试听扩展 vocal/bg；前端结果卡与历史卡新增「AI人声」「伴奏」tab + 分轨下载按钮（老批次自动隐藏）
- 价值：对齐竞品核心架构优势——用户可把 AI 人声导入自己的 DAW/声卡链做深度后期

### ② 方波质检 ✅ 本轮已落地（2026-11-15）
- 竞品口径：ffmpeg `astats`（1 秒帧，连续 2 帧 `Bit_depth≤1 ∧ Zero_crossings≤1 ∧ Entropy<0.15 ∧ Peak>-1 ∧ RMS>-6`）
- 实证差异：float 域下 astats 的 `Bit_depth` 是 popcount 口径，只能抓"恰 ±1.0"的极端损坏 → 自研泛化实现 `engine/qc_square.py`
- 自研口径（1 秒帧，前置 peak>-1dB ∧ RMS>-6dB，连续 2 帧坏帧 → 报起始秒）：
  - **A 卡死平台**：过零数≤2 ∧ 量化电平数≤4（含满幅直流与非满幅 0.99 卡死；不依赖饱和占比，防漏报）
  - **B 逐样本交替**：饱和占比≥0.95 ∧ 电平数≤2 ∧ (过零≤2 ∨ 过零≥0.8×帧样本数)
    （过零约束区分"损坏交替"与合法低频方波音乐——220Hz 方波过零≈440 不误报）
  - 单测 4/4：直流平台✅ 合法 220Hz 方波✅ 逐样本交替✅ 正常歌声✅；真实成品零误报
- 接入：stage4 后对 `converted.wav` + `mix.wav` 双检，命中写 `stats.qc` 与 `result.qc`，前端出红标「疑似损坏」——**只标记不拦截**
- CLI：`.venv/bin/python engine/qc_square.py <wav>...`

### ③ 训练档（本地微调专属音色）⏳ 未做，等口令
- 方向：本地 So-VITS-SVC / DDSP-SVC 微调（10–40 分钟干声 → 专属 checkpoint），与现有零样本档并存
- 回应竞品"效果更好"的根因；保留本地隐私卖点（竞品是云端训练）
- 工程量：新增训练子命令 + 模型管理 + 前端音色库；预估中大型迭代

## 三、明确不学的

- **VST 插件**（.dll/.vst3）：绑定 Windows 声卡生态，工程量大，与 Web 跨平台路线冲突
- **双设备路由**（监听/输出分离 + 命名管道 IPC）：场景窄（直播连麦），Web Audio 也能做但优先级极低
- **云端训练**：直接放弃隐私/离线卖点，正面对撞竞品优势区

## 四、其他可抄小件（顺手级）

- 调性检测（essentia.js / aubio 本地跑，自动建议移调量）——等口令
- 去和声开关、更细的 infer-mode 参数体系——SEED-VC 侧慢慢加

---

## 附：本轮改动落点（2026-11-15）

- `engine/qc_square.py`（新增）：方波检测器 + CLI
- `app.py`：stage4 分轨落盘 + 质检双检；`/result` 加 `kind`；`/audio` 加 vocal/bg；`/history` files 加 vocal/bg
- `static/index.html`：结果卡分轨 tab/下载/质检红标；历史卡分轨 tab/下载；补 `.mini` 按钮样式（原为裸链接）

> AI生成
