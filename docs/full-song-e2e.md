# 完整歌曲端到端验证报告（《贝加尔湖畔》）

## 结论
用「用户本人音色（我唱的童话.m4a）」成功完成《贝加尔湖畔》全曲翻唱，完整链路跑通。

## 输入
- 源歌：贝加尔湖畔.mp3（李健，284.17s，44.1kHz 立体声 mp3 203kbps）
- 参考音色：我唱的童话.m4a（13.44s，16kHz 单声道 AAC）

## 链路与耗时
| 阶段 | 耗时 | 说明 |
|---|---|---|
| Demucs(htdemucs) 分轨 | ~11s | 约 50 倍实时，MPS |
| SEED-VC 模型加载 | 8.06s | 加载后峰值 RSS 2.28GB |
| Whisper 语义特征 | 0.64s | 长音频内部分块 |
| RMVPE F0 提取 | 19.11s | 分块（30s + 1s 重叠） |
| Length regulator | 0.32s | |
| 扩散解码 + BigVGAN | 187.21s | 18 块，RTF 0.659 |
| SEED-VC 小计 | 207.71s | 峰值 RSS 2.50GB |
| 回混 + 转码 | ~2s | |
| **端到端总计** | **~3.7min** | 满足 ≤5min |

## 关键修复：长音频 MPS 共享内存崩溃
- 现象：SEED-VC 对 286s 人声提 F0 时 `Assertion failed: Failed to allocate IOGPUDeviceShmem` 崩溃（28s 短音频正常）。
- 定位：给 `convert()` 加逐阶段即时打印，崩溃点锁定在「Mel 之后、RMVPE F0 之前」——即 RMVPE 对 286s 音频一次性推理。
- 修复：`infer_svc.py` 新增 `rmvpe_infer_chunked()`，30s 分块 + 1s 重叠提取 F0 后拼接。`length_regulator` 会对 F0 做最近邻重采样到目标长度，分块拼接的微小边界误差无影响。

## 混音参数
- 人声（单声道）复制为居中双声道；人声增益 1.0、伴奏增益 0.9。
- 实测相加峰值 0.891，未削波，无需额外限幅。

## 产物
- engine/output/贝加尔湖畔_我的音色_AI.wav（50MB 无损）
- engine/output/贝加尔湖畔_我的音色_AI.mp3（6.8MB，192k）

## 用户四维初评（已反馈）
- 音色相似度：像本人 ✓
- 歌词可懂：吐词清晰 ✓
- 音高节奏：没跑调 ✓
- 伪影：未提及明显问题（整体正面）

> 结论：SEED-VC(singing checkpoint) + Demucs 技术路线端到端验证通过，可进入 M2 前端联调。
