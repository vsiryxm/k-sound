#!/usr/bin/env python3
"""k-sound · M1 推理脚本

SEED-VC 零样本歌声转换（singing checkpoint：f0-condition，44.1kHz，BigVGAN）。
在 Apple Silicon 上优先使用 MPS，带分阶段计时与峰值内存统计。
运行前确保已安装依赖，并以本文件所在目录作为参考（模块相对导入）。
"""

import argparse
import os
import resource
import sys
import time
import warnings
from contextlib import nullcontext

warnings.filterwarnings("ignore")

# 本机直连 huggingface.co 超时，统一走镜像；须在 import huggingface 相关库前设置。
# 关键：禁用 xet。hf-mirror 对 xet 存储的文件会 302 到 cas-bridge.xethub.hf.co（被墙），
# 关闭 xet 后走镜像的 LFS 代理(cdn-lfs.hf-mirror.com)才能正常下载。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch
import torchaudio
import librosa
import yaml

SEED_VC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed-vc")
sys.path.insert(0, SEED_VC_DIR)

from modules.commons import build_model, load_checkpoint, recursive_munch
from modules.campplus.DTDNN import CAMPPlus
from modules.bigvgan import bigvgan
from modules.audio import mel_spectrogram
from modules.rmvpe import RMVPE
from transformers import AutoFeatureExtractor, WhisperModel
from hf_utils import load_custom_model_from_hf


# singing checkpoint（f0-condition 歌声模型，44.1kHz）
SINGING_CHECKPOINT = "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth"
SINGING_CONFIG = "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"
CAMPLUS_LOCAL = os.path.join(SEED_VC_DIR, "campplus_cn_common.bin")


class StageTimer:
    """记录各阶段累计耗时。"""

    def __init__(self):
        self.start = time.time()
        self.records = []  # [(name, elapsed_seconds)]

    def mark(self, name):
        now = time.time()
        elapsed = now - self.start
        self.records.append((name, elapsed))
        self.start = now
        print(f"[stage] {name}  (+{elapsed:.2f}s)", flush=True)

    def report(self):
        total = sum(d for _, d in self.records)
        lines = []
        for name, d in self.records:
            lines.append(f"  - {name}: {d:.2f}s")
        lines.append(f"  - [总耗时] {total:.2f}s")
        return "\n".join(lines)


def peak_rss_mb():
    # macOS 上 ru_maxrss 单位为字节
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def mps_mem_mb():
    if torch.backends.mps.is_available():
        try:
            cur = torch.mps.current_allocated_memory() / (1024 * 1024)
            drv = torch.mps.driver_allocated_memory() / (1024 * 1024)
            return cur, drv
        except Exception:
            return None, None
    return None, None


def select_device(prefer_mps=True):
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_singing_models(device):
    dit_path, cfg_path = load_custom_model_from_hf(
        "Plachta/Seed-VC", SINGING_CHECKPOINT, SINGING_CONFIG
    )
    config = yaml.safe_load(open(cfg_path, "r"))
    model_params = recursive_munch(config["model_params"])
    model = build_model(model_params, stage="DiT")
    hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
    sr = config["preprocess_params"]["sr"]

    model, _, _, _ = load_checkpoint(
        model,
        None,
        dit_path,
        load_only_params=True,
        ignore_modules=[],
        is_distributed=False,
    )
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    # Whisper 语义编码器（仅编码器）
    whisper_name = model_params.speech_tokenizer.name
    whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(device)
    del whisper_model.decoder
    whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

    # 说话人风格编码器（使用仓库自带 campplus 权重）
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(CAMPLUS_LOCAL, map_location="cpu"))
    campplus_model.eval()
    campplus_model.to(device)

    # 44kHz BigVGAN 声码器
    bigvgan_model = bigvgan.BigVGAN.from_pretrained(
        "nvidia/bigvgan_v2_44khz_128band_512x", use_cuda_kernel=False
    )
    bigvgan_model.remove_weight_norm()
    bigvgan_model = bigvgan_model.eval().to(device)

    # F0 提取器
    rmvpe_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
    rmvpe = RMVPE(rmvpe_path, is_half=False, device=device)

    return (
        model,
        whisper_model,
        whisper_feature_extractor,
        campplus_model,
        bigvgan_model,
        rmvpe,
        config,
        sr,
        hop_length,
    )


def whisper_features(whisper_model, feature_extractor, audio_16k, device):
    """对 16kHz 波形提取 Whisper 语义特征，支持长音频分块（30s 窗口、5s 重叠）。"""
    if audio_16k.size(-1) <= 16000 * 30:
        inputs = feature_extractor(
            [audio_16k.squeeze(0).cpu().numpy()],
            return_tensors="pt",
            return_attention_mask=True,
            sampling_rate=16000,
        )
        input_features = whisper_model._mask_input_features(
            inputs.input_features, attention_mask=inputs.attention_mask
        ).to(device)
        outputs = whisper_model.encoder(
            input_features.to(whisper_model.encoder.dtype),
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        features = outputs.last_hidden_state.to(torch.float32)
        features = features[:, : audio_16k.size(-1) // 320 + 1]
        return features

    overlapping_time = 5
    features_list = []
    buffer = None
    traversed_time = 0
    while traversed_time < audio_16k.size(-1):
        if buffer is None:
            chunk = audio_16k[:, traversed_time : traversed_time + 16000 * 30]
        else:
            chunk = torch.cat(
                [
                    buffer,
                    audio_16k[:, traversed_time : traversed_time + 16000 * (30 - overlapping_time)],
                ],
                dim=-1,
            )
        inputs = feature_extractor(
            [chunk.squeeze(0).cpu().numpy()],
            return_tensors="pt",
            return_attention_mask=True,
            sampling_rate=16000,
        )
        input_features = whisper_model._mask_input_features(
            inputs.input_features, attention_mask=inputs.attention_mask
        ).to(device)
        outputs = whisper_model.encoder(
            input_features.to(whisper_model.encoder.dtype),
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        chunk_features = outputs.last_hidden_state.to(torch.float32)
        chunk_features = chunk_features[:, : chunk.size(-1) // 320 + 1]
        if traversed_time == 0:
            features_list.append(chunk_features)
        else:
            features_list.append(chunk_features[:, 50 * overlapping_time :])
        buffer = chunk[:, -16000 * overlapping_time :]
        traversed_time += (
            30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
        )
    return torch.cat(features_list, dim=1)


def adjust_f0_semitones(f0_sequence, n_semitones):
    factor = 2 ** (n_semitones / 12)
    return f0_sequence * factor


def crossfade(chunk1, chunk2, overlap):
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    if len(chunk2) < overlap:
        chunk2[:overlap] = (
            chunk2[:overlap] * fade_in[: len(chunk2)]
            + (chunk1[-overlap:] * fade_out)[: len(chunk2)]
        )
    else:
        chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


def rmvpe_infer_chunked(rmvpe, audio, chunk_sec=30, overlap_sec=1, thred=0.03):
    """对长音频分块提取 F0，规避 MPS 对超长序列一次性分配 shmem 失败的问题。

    RMVPE 在 16kHz 下帧移 hop=160；重叠区按对应帧数丢弃。length_regulator
    会对 F0 做最近邻重采样到目标长度，因此分块拼接的微小边界误差无影响。
    """
    hop = 160
    chunk_len = int(chunk_sec * 16000)
    overlap_len = int(overlap_sec * 16000)
    drop = int(round(overlap_len / hop))

    if not torch.is_tensor(audio):
        audio = torch.from_numpy(audio)
    n = audio.shape[-1]
    if n <= chunk_len:
        return np.asarray(rmvpe.infer_from_audio(audio, thred=thred))

    pieces = []
    pos = 0
    while pos < n:
        chunk = audio[pos : pos + chunk_len]
        f0 = np.asarray(rmvpe.infer_from_audio(chunk, thred=thred))
        if pos == 0:
            pieces.append(f0)
        else:
            pieces.append(f0[drop:])
        pos += chunk_len - overlap_len
    return np.concatenate(pieces)


@torch.no_grad()
def convert(
    model,
    whisper_model,
    feature_extractor,
    campplus_model,
    bigvgan_model,
    rmvpe,
    sr,
    hop_length,
    source_path,
    target_path,
    diffusion_steps=30,
    length_adjust=1.0,
    inference_cfg_rate=0.7,
    auto_f0_adjust=False,
    pitch_shift=0,
    fp16=False,
    device=None,
):
    timer = StageTimer()
    max_context_window = sr // hop_length * 30
    overlap_frame_len = 16
    overlap_wave_len = overlap_frame_len * hop_length

    # 读取音频并重采样到模型采样率
    source_audio = librosa.load(source_path, sr=sr)[0]
    ref_audio = librosa.load(target_path, sr=sr)[0]
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(ref_audio[: sr * 25]).unsqueeze(0).float().to(device)
    timer.mark("加载音频(重采样至44.1kHz)")

    # 重采样到 16kHz 供 Whisper 与 F0
    ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
    timer.mark("重采样至16kHz")

    S_alt = whisper_features(whisper_model, feature_extractor, converted_waves_16k, device)
    S_ori = whisper_features(whisper_model, feature_extractor, ref_waves_16k, device)
    timer.mark("Whisper 语义特征提取")

    mel = mel_spectrogram(
        source_audio.to(device).float(),
        n_fft=2048,
        win_size=2048,
        hop_size=hop_length,
        num_mels=128,
        sampling_rate=sr,
        fmin=0,
        fmax=None,
        center=False,
    )
    mel2 = mel_spectrogram(
        ref_audio.to(device).float(),
        n_fft=2048,
        win_size=2048,
        hop_size=hop_length,
        num_mels=128,
        sampling_rate=sr,
        fmin=0,
        fmax=None,
        center=False,
    )

    timer.mark("Mel 频谱")

    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    feat2 = torchaudio.compliance.kaldi.fbank(
        ref_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000
    )
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    # F0 提取与音高对齐
    F0_ori = rmvpe.infer_from_audio(ref_waves_16k[0], thred=0.03)
    F0_alt = rmvpe_infer_chunked(rmvpe, converted_waves_16k[0], thred=0.03)
    F0_ori = torch.from_numpy(F0_ori).float().to(device)[None]
    F0_alt = torch.from_numpy(F0_alt).float().to(device)[None]

    voiced_F0_ori = F0_ori[F0_ori > 1]
    voiced_F0_alt = F0_alt[F0_alt > 1]
    log_f0_alt = torch.log(F0_alt + 1e-5)
    voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
    voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
    median_log_f0_ori = torch.median(voiced_log_f0_ori)
    median_log_f0_alt = torch.median(voiced_log_f0_alt)

    shifted_log_f0_alt = log_f0_alt.clone()
    if auto_f0_adjust:
        shifted_log_f0_alt[F0_alt > 1] = (
            log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
        )
    shifted_f0_alt = torch.exp(shifted_log_f0_alt)
    if pitch_shift != 0:
        shifted_f0_alt[F0_alt > 1] = adjust_f0_semitones(
            shifted_f0_alt[F0_alt > 1], pitch_shift
        )
    timer.mark("RMVPE F0 提取与音高对齐")

    # Length regulation
    cond, _, _, _, _ = model.length_regulator(
        S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt
    )
    prompt_condition, _, _, _, _ = model.length_regulator(
        S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori
    )
    timer.mark("Length regulator")

    max_source_window = max_context_window - mel2.size(2)
    processed_frames = 0
    generated_wave_chunks = []
    previous_chunk = None
    chunk_count = 0
    vc_start = time.time()
    # 供外部进度解析：预估总块数
    total_chunks = max(1, (cond.size(1) + max_source_window - 1) // max_source_window)

    while processed_frames < cond.size(1):
        chunk_cond = cond[:, processed_frames : processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)

        # MPS 不支持 torch.autocast（torch 2.4 会报 unsupported device_type 'mps'），
        # 仅在 CUDA 上启用 fp16 autocast；MPS/CPU 直接以 float32 原生推理。
        if device.type == "cuda" and fp16:
            amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
        else:
            amp_ctx = nullcontext()

        with amp_ctx:
            vc_target = model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                mel2,
                style2,
                None,
                diffusion_steps,
                inference_cfg_rate=inference_cfg_rate,
            )
            vc_target = vc_target[:, :, mel2.size(-1):]

        vc_wave = bigvgan_model(vc_target.float()).squeeze()
        vc_wave = vc_wave[None, :]
        chunk_count += 1
        print(f"[chunk] {chunk_count}/{total_chunks}", flush=True)

        if processed_frames == 0:
            if is_last_chunk:
                output_wave = vc_wave[0].cpu().numpy()
                generated_wave_chunks.append(output_wave)
                break
            output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
        elif is_last_chunk:
            output_wave = crossfade(
                previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len
            )
            generated_wave_chunks.append(output_wave)
            processed_frames += vc_target.size(2) - overlap_frame_len
            break
        else:
            output_wave = crossfade(
                previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(),
                overlap_wave_len,
            )
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len

    vc_wave = torch.tensor(np.concatenate(generated_wave_chunks))[None, :].float()
    vc_elapsed = time.time() - vc_start
    timer.mark(f"扩散解码 + BigVGAN({chunk_count} 块)")
    print(f"[perf] 扩散/BigVGAN 阶段 RTF: {vc_elapsed / (vc_wave.size(-1) / sr):.3f}")

    return vc_wave, sr, timer, chunk_count


def main():
    parser = argparse.ArgumentParser(description="k-sound M1 SEED-VC singing conversion")
    parser.add_argument("--source", type=str, required=True, help="歌曲人声（源，将保留其旋律/内容）")
    parser.add_argument("--target", type=str, required=True, help="参考音色（目标声音）")
    parser.add_argument("--output", type=str, required=True, help="输出 wav 路径")
    parser.add_argument("--diffusion-steps", type=int, default=30)
    parser.add_argument("--length-adjust", type=float, default=1.0)
    parser.add_argument("--inference-cfg-rate", type=float, default=0.7)
    parser.add_argument("--auto-f0-adjust", action="store_true")
    parser.add_argument("--pitch-shift", type=int, default=0)
    parser.add_argument("--fp16", action="store_true", help="扩散阶段用 float16 autocast（默认 float32）")
    parser.add_argument("--cpu", action="store_true", help="强制 CPU（默认优先 MPS）")
    args = parser.parse_args()

    device = select_device(prefer_mps=not args.cpu)
    print(f"[device] 使用 {device}")

    load_start = time.time()
    models = load_singing_models(device)
    (
        model,
        whisper_model,
        feature_extractor,
        campplus_model,
        bigvgan_model,
        rmvpe,
        config,
        sr,
        hop_length,
    ) = models
    load_elapsed = time.time() - load_start
    cur, drv = mps_mem_mb()
    print(f"[perf] 模型加载耗时: {load_elapsed:.2f}s")
    print(f"[perf] 加载后峰值 RSS: {peak_rss_mb():.1f} MB")
    if cur is not None:
        print(f"[perf] MPS 当前/驱动分配内存: {cur:.1f}/{drv:.1f} MB")

    vc_wave, sr, timer, chunk_count = convert(
        model,
        whisper_model,
        feature_extractor,
        campplus_model,
        bigvgan_model,
        rmvpe,
        sr,
        hop_length,
        args.source,
        args.target,
        diffusion_steps=args.diffusion_steps,
        length_adjust=args.length_adjust,
        inference_cfg_rate=args.inference_cfg_rate,
        auto_f0_adjust=args.auto_f0_adjust,
        pitch_shift=args.pitch_shift,
        fp16=args.fp16,
        device=device,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torchaudio.save(args.output, vc_wave.cpu(), sr)
    out_dur = vc_wave.size(-1) / sr
    print(f"[perf] 输出时长: {out_dur:.2f}s")
    print("[perf] 分阶段耗时:")
    print(timer.report())
    print(f"[perf] 峰值 RSS(进程): {peak_rss_mb():.1f} MB")
    cur, drv = mps_mem_mb()
    if cur is not None:
        print(f"[perf] MPS 当前/驱动分配内存: {cur:.1f}/{drv:.1f} MB")
    print(f"[done] {args.output}")


if __name__ == "__main__":
    main()
