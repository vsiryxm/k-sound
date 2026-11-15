#!/usr/bin/env python3
"""k-sound · M2 Web 服务（FastAPI，本机 localhost DEMO）

任务式接口：
  POST /jobs             上传 voice(本人干声) + song(歌曲)，创建翻唱任务
  GET  /jobs/{id}        查询任务阶段 / 状态 / 错误
  GET  /jobs/{id}/result 下载成品（?fmt=mp3|wav，?kind=mix|vocal|bg 分轨下载）
  GET  /jobs/{id}/audio/{kind}  对比试听（cover|song|voice|vocal|bg，切换对比用）
  GET  /history          翻唱历史列表（SQLite k-sound.db，含速度/内存统计）
  GET|POST /api/config   生成设置（质量/音高，默认全本地模型）
  GET  /                 前端单页

链路（沿用 M1 已验证流程，全部以子进程调用、互相隔离）：
  1) ffmpeg 预处理：干声→44.1kHz mono wav；歌曲→44.1kHz stereo wav
  2) demucs.separate -n htdemucs --two-stems vocals 分轨
  3) engine/infer_svc.py SEED-VC 换音色（singing checkpoint，解析 [chunk] 块进度）
  4) 回混：人声(单声道 float32，勿再除 32768)复制双声道×1.0 + 伴奏×0.9，
     峰值>1 整体归一到 0.95 → 44.1kHz/16bit wav → 192k mp3，成品带 _AI 后缀
     分轨导出（对标竞品 bg/ai 双轨）：ai.wav=处理后的 AI 人声、bg.wav=Demucs 伴奏，
     可下载后导入 DAW/声卡效果链做深度后期
     方波质检（对标竞品 astats 质检）：converted/mix 双检，命中只标记不拦截

关键坑（沿用 M1 结论）：
  - HF 子进程必须带 HF_ENDPOINT=https://hf-mirror.com + HF_HUB_DISABLE_XET=1
  - infer_svc.py 的 checkpoint 缓存是 cwd 相对路径 ./checkpoints → 子进程 cwd 必须是项目根
  - MPS 不用 autocast（infer_svc.py 已内置）
"""

import json
import os
import queue
import re
import shutil
import sqlite3
import subprocess
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pedalboard import Compressor, HighpassFilter, Pedalboard, Reverb

from engine.qc_square import detect_square_wave

ROOT = Path(__file__).resolve().parent
JOBS_DIR = ROOT / "jobs"
ENGINE_SCRIPT = ROOT / "engine" / "infer_svc.py"
VENV_PY = ROOT / ".venv" / "bin" / "python"
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_QUEUE = 3

STAGE_LABELS = {
    "preprocess": "准备音频",
    "demucs": "分离人声",
    "svc": "音色转换",
    "remix": "混音导出",
}

app = FastAPI(title="k-sound", docs_url=None, redoc_url=None)

# ---------------------------------------------------------------- 任务管理

_jobs_lock = threading.Lock()
JOBS: dict = {}           # job_id -> job dict（内存态，同步持久化到磁盘）
JOB_QUEUE: "queue.Queue[str]" = queue.Queue()

_persist_lock = threading.Lock()


def _now() -> float:
    return time.time()


def _persist(job: dict) -> None:
    """把任务状态写进 jobs/<id>/status.json（原子写）。"""
    try:
        with _persist_lock:
            path = Path(job["dir"]) / "status.json"
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(path)
    except Exception:
        pass


# ---------------------------------------------------------------- SQLite（翻唱历史 + 生成设置）

DB_PATH = ROOT / "k-sound.db"

QUALITY_STEPS = {"fast": 20, "standard": 30, "fine": 45}
QUALITY_LABEL = {"fast": "快速", "standard": "标准", "fine": "精细"}
DEFAULT_CONFIG = {"quality": "standard", "pitch_shift": 0, "auto_f0": False, "reverb_wet": 0.18}

_AUDIO_MEDIA = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
                ".flac": "audio/flac", ".aac": "audio/aac", ".ogg": "audio/ogg"}

_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY,
  created_at REAL,
  voice_name TEXT DEFAULT '',
  song_name TEXT DEFAULT '',
  song_duration REAL DEFAULT 0,
  status TEXT DEFAULT '',
  error TEXT DEFAULT '',
  total_s REAL, speed_x REAL, model_load_s REAL,
  peak_rss_gb REAL, demucs_peak_rss_gb REAL,
  mps_current_gb REAL, mps_driver_gb REAL, svc_rtf REAL,
  diffusion_steps INTEGER,
  pitch_shift INTEGER DEFAULT 0, auto_f0 INTEGER DEFAULT 0,
  reverb_wet REAL DEFAULT 0,
  stage_times TEXT DEFAULT '{}',
  result_filename TEXT DEFAULT '',
  result_duration REAL, result_size_bytes INTEGER,
  finished_at REAL
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _db_init() -> None:
    conn = _db_conn()
    try:
        with conn:
            conn.executescript(_DB_SCHEMA)
            # 旧库迁移：补 reverb_wet 列（幂等）
            cols = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
            if "reverb_wet" not in cols:
                conn.execute("ALTER TABLE jobs ADD COLUMN reverb_wet REAL DEFAULT 0")
    finally:
        conn.close()


def _db_exec(query: str, params: tuple = ()) -> None:
    conn = _db_conn()
    try:
        with conn:
            conn.execute(query, params)
    finally:
        conn.close()


def _db_query(query: str, params: tuple = ()) -> list:
    conn = _db_conn()
    try:
        return conn.execute(query, params).fetchall()
    finally:
        conn.close()


def _db_upsert_job(job: dict) -> None:
    """任务终态（done/error）写入历史库；历史库异常不影响主流程。"""
    s = job.get("stats") or {}
    r = job.get("result") or {}
    cfg = job.get("settings_used") or {}
    try:
        _db_exec(
            "INSERT OR REPLACE INTO jobs (job_id, created_at, voice_name, song_name, "
            "song_duration, status, error, total_s, speed_x, model_load_s, peak_rss_gb, "
            "demucs_peak_rss_gb, mps_current_gb, mps_driver_gb, svc_rtf, diffusion_steps, "
            "pitch_shift, auto_f0, reverb_wet, stage_times, result_filename, result_duration, "
            "result_size_bytes, finished_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (job.get("job_id"), job.get("created_at"), job.get("voice_name", ""),
             job.get("song_name", ""), job.get("song_duration") or 0,
             job.get("status", ""), job.get("error", ""),
             s.get("total_s"), s.get("speed_x"), s.get("model_load_s"),
             s.get("peak_rss_gb"), s.get("demucs_peak_rss_gb"),
             s.get("mps_current_gb"), s.get("mps_driver_gb"), s.get("svc_rtf"),
             cfg.get("diffusion_steps", s.get("diffusion_steps")),
             cfg.get("pitch_shift", 0), 1 if cfg.get("auto_f0") else 0,
             s.get("reverb_wet", 0),
             json.dumps(s.get("stage_times") or {}, ensure_ascii=False),
             r.get("filename", ""), r.get("duration"), r.get("size_bytes"),
             job.get("finished_at")))
    except sqlite3.Error:
        pass


def _get_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    try:
        for row in _db_query("SELECT key, value FROM settings"):
            cfg[row["key"]] = json.loads(row["value"])
    except (sqlite3.Error, json.JSONDecodeError):
        pass
    if cfg.get("quality") not in QUALITY_STEPS:
        cfg["quality"] = "standard"
    try:
        cfg["pitch_shift"] = int(cfg.get("pitch_shift") or 0)
    except (TypeError, ValueError):
        cfg["pitch_shift"] = 0
    cfg["pitch_shift"] = max(-12, min(12, cfg["pitch_shift"]))
    cfg["auto_f0"] = bool(cfg.get("auto_f0"))
    try:
        wet = round(float(cfg.get("reverb_wet") or 0.0), 2)
    except (TypeError, ValueError):
        wet = 0.0
    cfg["reverb_wet"] = max(0.0, min(0.3, wet))
    return cfg


def _save_config(cfg: dict) -> None:
    for key, value in cfg.items():
        _db_exec("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                 (key, json.dumps(value, ensure_ascii=False)))


def _new_job(voice_name: str, song_name: str, song_duration: float) -> dict:
    job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    job_dir = JOBS_DIR / job_id
    (job_dir / "upload").mkdir(parents=True, exist_ok=True)
    job = {
        "job_id": job_id,
        "dir": str(job_dir),
        "status": "queued",
        "stage": None,
        "detail": "",
        "queue_position": None,
        "error": "",
        "error_debug": "",
        "voice_name": voice_name,
        "song_name": song_name,
        "song_duration": song_duration,
        "created_at": _now(),
        "started_at": None,
        "finished_at": None,
        "result": None,
    }
    with _jobs_lock:
        JOBS[job_id] = job
    _persist(job)
    return job


def _set(job: dict, **kw) -> None:
    with _jobs_lock:
        job.update(kw)
    _persist(job)


def _load_persisted() -> None:
    """启动恢复：读取磁盘 status.json；中断的任务标记为 error。"""
    if not JOBS_DIR.exists():
        return
    for path in sorted(JOBS_DIR.glob("*/status.json")):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if job.get("status") in ("queued", "running"):
            job["status"] = "error"
            job["error"] = "服务重启，任务已中断，请重新生成。"
            job["finished_at"] = _now()
            path.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
        with _jobs_lock:
            JOBS[job["job_id"]] = job
        _db_upsert_job(job)


_db_init()
_load_persisted()


def _public(job: dict) -> dict:
    out = {k: job.get(k) for k in (
        "job_id", "status", "stage", "detail", "error",
        "voice_name", "song_name", "song_duration",
        "created_at", "started_at", "finished_at", "result", "stats",
    )}
    out["batch_id"] = job.get("job_id")  # 批次号：一次生成操作 = 一个批次（含干声/源歌/翻唱三件）
    out["stage_label"] = STAGE_LABELS.get(job.get("stage"), "")
    if job.get("status") == "queued":
        out["queue_position"] = JOB_QUEUE.qsize()
    return out


# ---------------------------------------------------------------- 子进程工具

HF_HOME = os.path.expanduser("~/.cache/huggingface")


def _child_env() -> dict:
    """推理子进程环境。

    关键：本 Agent 运行环境自带 HF_HOME=~/.cache/TeleAgent/huggingface（另一份空缓存），
    会害子进程找不到 whisper-small / bigvgan —— 必须显式指回真实缓存 ~/.cache/huggingface。
    所有模型均已缓存，加 HF_HUB_OFFLINE=1 纯离线解析：运行期零联网、零代理依赖、启动更快。
    另剔除代理变量，避免子进程尝试不可达的直连。
    """
    env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["HF_HUB_DISABLE_XET"] = "1"
    env["HF_HOME"] = HF_HOME
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run(cmd: list, cwd: Path, timeout: int, on_line=None, tail_lines: int = 40) -> tuple:
    """运行子进程，逐行读合并输出；返回 (returncode, output_tail)。
    子进程独立进程组（start_new_session），看门狗超时按组 SIGKILL，
    连孙进程一起清掉，避免包装脚本/孙进程变孤儿；返回码 124 = 超时。"""
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), env=_child_env(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, start_new_session=True,
    )
    lines: list = []
    timed_out = threading.Event()

    def _kill_tree() -> None:
        timed_out.set()
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    watchdog = threading.Timer(timeout, _kill_tree)
    watchdog.start()
    try:
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            lines.append(line)
            if len(lines) > 400:
                del lines[:200]
            if on_line:
                on_line(line)
        code = proc.wait(timeout=60)
    except Exception:
        _kill_tree()
        raise
    finally:
        watchdog.cancel()
    if timed_out.is_set():
        return 124, "\n".join(lines[-tail_lines:])
    return (code if code is not None else 1), "\n".join(lines[-tail_lines:])


def _parse_perf(line: str, perf: dict) -> None:
    """解析 infer_svc.py 的 [perf] 行 → 性能统计（供成本估算）。"""
    m = re.search(r"扩散/BigVGAN 阶段 RTF:\s*([\d.]+)", line)
    if m:
        perf["svc_rtf"] = round(float(m.group(1)), 3)
    m = re.search(r"模型加载耗时:\s*([\d.]+)s", line)
    if m:
        perf["model_load_s"] = round(float(m.group(1)), 1)
    m = re.search(r"峰值 RSS\(进程\):\s*([\d.]+)\s*MB", line)
    if m:
        perf["peak_rss_mb"] = float(m.group(1))
    m = re.search(r"MPS 当前/驱动分配内存:\s*([\d.]+)/([\d.]+)\s*MB", line)
    if m:
        perf["mps_current_mb"] = float(m.group(1))
        perf["mps_driver_mb"] = float(m.group(2))


def _ffprobe_duration(path: Path) -> float:
    try:
        r = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            raise ValueError(r.stderr[-300:])
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        raise ValueError("无法解码音频文件")


def _volume_db(path: Path) -> float | None:
    """ffmpeg volumedetect 的 mean volume（dB）。"""
    try:
        r = subprocess.run(
            [FFMPEG, "-hide_banner", "-nostats", "-i", str(path),
             "-af", "volumedetect", "-f", "null", "-"],
            capture_output=True, text=True, timeout=180,
        )
        m = re.search(r"mean_volume:\s*(-?[\d.]+)\s*dB", r.stderr)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def _qc_check(path) -> dict | None:
    """方波质检；任何异常都按"未检出"处理，绝不影响主流程。
    返回 None（正常）或 {"start_sec": 起始秒}。"""
    try:
        return detect_square_wave(path)
    except Exception:
        return None


# ---------------------------------------------------------------- 工作线程

def _worker() -> None:
    while True:
        job_id = JOB_QUEUE.get()
        try:
            _run_job(JOBS.get(job_id) or job_id)
        except Exception as exc:  # 兜底，不该走到
            job = JOBS.get(job_id)
            if job:
                _set(job, status="error", error=f"生成失败：{exc}",
                     error_debug=str(exc), finished_at=_now())
        finally:
            JOB_QUEUE.task_done()


threading.Thread(target=_worker, daemon=True, name="ks-worker").start()


def _process_vocal(vocal_st: np.ndarray, wet: float, gain: float = 0.90) -> np.ndarray:
    """人声处理链（融合度优化 B 步）：把 AI 人声"坐进"伴奏声场。

    链路：HP高通去泥 → 轻压缩控动态 → 混响(带pre-delay) → Haas 微展宽 → 增益
    - wet=0 时跳过混响（其余链路保留）
    - pedalboard 全本地 DSP，处理耗时 <1 秒
    """
    x = vocal_st.astype(np.float32)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    board_plugins = [
        HighpassFilter(cutoff_frequency_hz=90),        # 去低频泥，给人声清位
        Compressor(threshold_db=-18, ratio=2.5, attack_ms=8, release_ms=160),
        # 轻压缩：动态贴合乐队律动（压缩损失 ~0.6dB，统一在 gain 里补）
    ]
    if wet > 0:
        board_plugins.append(Reverb(room_size=0.45, damping=0.62, wet_level=wet,
                                     dry_level=1.0, width=1.0, freeze_mode=0.0))
    board = Pedalboard(board_plugins)
    out = board(x, 44100)
    # Haas 微展宽：右声道延迟 ~6ms，消除正中央点声源感（幅度控制在人耳定位阈内）
    d = int(0.006 * 44100)
    out = out.copy()
    out[d:, 1] = out[:-d, 1] if len(out) > d else out[:, 1]
    out[:d, 1] = 0
    return out * gain


def _upload_file(job_dir: Path, kind: str) -> Path | None:
    """上传件统一入口：新任务保留原始扩展名（voice.m4a / song.mp3），
    老任务回退无扩展名（voice / song）。返回第一个存在的文件，无则 None。"""
    upload = job_dir / "upload"
    for p in sorted(upload.glob(f"{kind}.*")):
        if p.is_file():
            return p
    legacy = upload / kind
    return legacy if legacy.exists() else None


def _run_job(job) -> None:
    if isinstance(job, str):
        job = JOBS[job]
    job_dir = Path(job["dir"])
    cfg = _get_config()
    steps = QUALITY_STEPS[cfg["quality"]]
    _set(job, status="running", started_at=_now(), queue_position=None,
         settings_used={"quality": cfg["quality"], "diffusion_steps": steps,
                        "pitch_shift": cfg["pitch_shift"], "auto_f0": cfg["auto_f0"],
                        "reverb_wet": cfg["reverb_wet"]})
    perf: dict = {}        # infer_svc / demucs 包装器输出的性能指标
    stage_times: dict = {}  # 各阶段墙钟耗时（秒）

    # ---- stage 1 · 预处理：统一转 44.1kHz wav ----
    t_stage = time.time()
    _set(job, stage="preprocess", detail="正在规范化音频格式")
    voice_wav = job_dir / "voice.wav"
    song_wav = job_dir / "song.wav"
    code, tail = _run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(_upload_file(job_dir, "voice")), "-ac", "1", "-ar", "44100",
         # SEED-VC 中参考干声会占用转换窗口（30s 上下文减去参考时长），
         # 参考越长分块越多、转换越慢；15s 是音色稳定与速度的平衡点
         # （官方建议 5–10s，13.4s 实测四维初评通过）
         "-t", "15", "-sample_fmt", "s16", str(voice_wav)],
        cwd=ROOT, timeout=120,
    )
    if code != 0 or not voice_wav.exists():
        _fail(job, "干声预处理失败（音频解码出错）", tail)
        return
    code, tail = _run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(_upload_file(job_dir, "song")), "-ac", "2", "-ar", "44100",
         "-sample_fmt", "s16", str(song_wav)],
        cwd=ROOT, timeout=180,
    )
    if code != 0 or not song_wav.exists():
        _fail(job, "歌曲预处理失败（音频解码出错）", tail)
        return
    stage_times["preprocess"] = round(time.time() - t_stage, 1)

    # ---- stage 2 · Demucs 分轨（包装器测峰值内存） ----
    t_stage = time.time()
    _set(job, stage="demucs", detail="正在分离人声与伴奏")

    def demucs_on_line(line: str) -> None:
        m = re.match(r"\[peak_rss\]\s*([\d.]+)\s*MB", line)
        if m:
            perf["demucs_peak_rss_mb"] = float(m.group(1))

    code, tail = _run(
        [str(VENV_PY), str(ROOT / "engine" / "run_with_peak.py"),
         str(VENV_PY), "-m", "demucs.separate", "-n", "htdemucs",
         "--two-stems", "vocals", "-o", str(job_dir), str(song_wav)],
        cwd=ROOT, timeout=900, on_line=demucs_on_line,
    )
    stage_times["demucs"] = round(time.time() - t_stage, 1)
    vocals_wav = job_dir / "htdemucs" / "song" / "vocals.wav"
    novocals_wav = job_dir / "htdemucs" / "song" / "no_vocals.wav"
    if code != 0 or not vocals_wav.exists() or not novocals_wav.exists():
        _fail(job, "人声分离失败（歌曲可能是纯音乐 / 现场录音 / 质量过低）", tail)
        return
    try:
        vocals, _ = sf.read(str(vocals_wav))
        rms = float(np.sqrt(np.mean(np.square(vocals.astype(np.float64)))))
        if 20 * np.log10(max(rms, 1e-12)) < -45:
            _fail(job, "人声分离后未检测到人声（可能是纯音乐，请更换单人主唱的歌曲）", "")
            return
    except Exception as exc:
        _fail(job, "人声分离结果读取失败", str(exc))
        return

    # ---- stage 3 · SEED-VC 换音色 ----
    t_stage = time.time()
    _set(job, stage="svc", detail="正在加载模型")

    def on_line(line: str) -> None:
        m = re.match(r"\[chunk\]\s*(\d+)/(\d+)", line)
        if m:
            _set(job, detail=f"已处理 {m.group(1)}/{m.group(2)} 段")
        elif line.startswith("[stage]"):
            name = re.sub(r"\s*\(\+[\d.]+s\)\s*$", "", line[7:]).strip()
            _set(job, detail=name or "处理中")
        else:
            if "模型加载耗时" in line:
                _set(job, detail="模型加载完成，开始转换")
            _parse_perf(line, perf)

    converted_wav = job_dir / "converted.wav"
    svc_cmd = [str(VENV_PY), str(ENGINE_SCRIPT),
               "--source", str(vocals_wav), "--target", str(voice_wav),
               "--output", str(converted_wav),
               "--diffusion-steps", str(steps),
               "--pitch-shift", str(cfg["pitch_shift"])]
    if cfg["auto_f0"]:
        svc_cmd.append("--auto-f0-adjust")
    code, tail = _run(svc_cmd, cwd=ROOT, timeout=1800, on_line=on_line, tail_lines=60)
    stage_times["svc"] = round(time.time() - t_stage, 1)
    if code != 0 or not converted_wav.exists():
        friendly = "音色转换失败"
        if code == 124:
            friendly = "音色转换超时，请尝试较短的歌曲"
        elif "IOGPUDeviceShmem" in tail or "Failed to allocate" in tail:
            friendly = "生成时内存不足，请尝试较短的歌曲后重试"
        _fail(job, friendly, tail)
        return

    # ---- stage 4 · 回混 + 导出 ----
    t_stage = time.time()
    _set(job, stage="remix", detail="正在混音导出")
    try:
        vocal, sr1 = sf.read(str(converted_wav))          # float32 单声道 [-1,1]，勿除 32768
        accomp, sr2 = sf.read(str(novocals_wav))           # 立体声 float
        if sr1 != 44100 or sr2 != 44100:
            raise RuntimeError(f"采样率异常: {sr1}/{sr2}")
        vocal = np.asarray(vocal).astype(np.float64)
        accomp = np.asarray(accomp).astype(np.float64)
        if accomp.ndim == 1:
            accomp = np.stack([accomp, accomp], axis=1)
        n = min(len(vocal), len(accomp))
        vocal_st = np.stack([vocal[:n], vocal[:n]], axis=1)
        vocal_st = _process_vocal(vocal_st, cfg["reverb_wet"])
        # D 步（融合度优化第一步）：人声 -1.5dB / 伴奏 +0.9dB
        # 人声从"骑在伴奏上"回到"坐在音乐里"
        mix = vocal_st * 0.85 + accomp[:n] * 1.0
        peak = float(np.abs(mix).max())
        if peak > 1.0:
            mix = mix * (0.95 / peak)
        mix_wav = job_dir / "mix.wav"
        sf.write(str(mix_wav), mix, 44100, subtype="PCM_16")
        # 分轨导出（对标竞品 bg/ai 双轨）：ai=处理后的 AI 人声、bg=Demucs 伴奏。
        # 峰值>1 时同样归一到 0.95，避免 PCM_16 写入时削波；分轨供 DAW/声卡链后期。
        ai_wav = job_dir / "ai.wav"
        bg_wav = job_dir / "bg.wav"
        for arr, path in ((vocal_st, ai_wav), (accomp[:n], bg_wav)):
            p = float(np.abs(arr).max())
            if p > 1.0:
                arr = arr * (0.95 / p)
            sf.write(str(path), arr, 44100, subtype="PCM_16")
    except Exception as exc:
        _fail(job, "混音失败", str(exc))
        return

    # 方波质检（对标竞品 astats 质检思路）：转换源 + 成品双检，命中只标记不拦截
    qc_converted = _qc_check(converted_wav)
    qc_mix = _qc_check(mix_wav)
    qc_hit = qc_mix or qc_converted   # 优先报成品（用户直接听到的那条）

    stem = _safe_stem(job["song_name"])
    mp3_path = job_dir / f"{stem}_AI.mp3"
    code, tail = _run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(mix_wav), "-codec:a", "libmp3lame", "-b:a", "192k",
         str(mp3_path)],
        cwd=ROOT, timeout=120,
    )
    if code != 0 or not mp3_path.exists():
        _fail(job, "MP3 导出失败", tail)
        return
    stage_times["remix"] = round(time.time() - t_stage, 1)

    duration = n / 44100.0
    total_s = round(time.time() - job["started_at"], 1)
    stats = {
        "total_s": total_s,
        "stage_times": dict(stage_times),
        "speed_x": round(duration / total_s, 2) if total_s > 0 else None,
        "model_load_s": perf.get("model_load_s"),
        "peak_rss_gb": round(perf["peak_rss_mb"] / 1024, 2) if "peak_rss_mb" in perf else None,
        "demucs_peak_rss_gb": round(perf["demucs_peak_rss_mb"] / 1024, 2) if "demucs_peak_rss_mb" in perf else None,
        "mps_current_gb": round(perf["mps_current_mb"] / 1024, 2) if "mps_current_mb" in perf else None,
        "mps_driver_gb": round(perf["mps_driver_mb"] / 1024, 2) if "mps_driver_mb" in perf else None,
        "svc_rtf": perf.get("svc_rtf"),
        "diffusion_steps": steps,
        "reverb_wet": cfg["reverb_wet"],
        "qc": {"converted": qc_converted, "mix": qc_mix},
    }
    result = {
        "filename": f"{stem}_AI.mp3",
        "duration": round(duration, 1),
        "size_bytes": mp3_path.stat().st_size,
        "wav_size_bytes": mix_wav.stat().st_size,
    }
    if ai_wav.exists() and bg_wav.exists():
        result["stems"] = {
            "vocal": "ai.wav", "bg": "bg.wav",
            "vocal_size_bytes": ai_wav.stat().st_size,
            "bg_size_bytes": bg_wav.stat().st_size,
        }
    if qc_hit:
        result["qc"] = qc_hit
    _set(job, status="done", stage=None, detail="", finished_at=_now(),
         stats=stats,
         result=result)
    _db_upsert_job(job)


def _fail(job: dict, friendly: str, debug: str) -> None:
    _set(job, status="error", stage=None, error=friendly,
         error_debug=(debug or "")[-2000:], finished_at=_now())
    _db_upsert_job(job)


def _safe_stem(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[\\/:*?\"<>|\x00-\x1f]", "_", stem).strip() or "output"
    return stem[:80]


# ---------------------------------------------------------------- API

def _check_audio_file(f: UploadFile, label: str) -> None:
    ext = Path(f.filename or "").suffix.lower()
    if ext not in AUDIO_EXTS:
        raise HTTPException(
            400, f"{label}格式不支持（{ext or '未知'}），请上传 mp3 / wav / m4a / flac 文件")
    if f.size is not None and f.size > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{label}文件过大（上限 500MB）")


@app.post("/jobs", status_code=202)
async def create_job(
    voice: UploadFile = File(...),
    song: UploadFile = File(...),
    authorized: str = Form("false"),
):
    if (authorized or "").strip().lower() not in ("true", "1", "yes"):
        raise HTTPException(400, "请先勾选「音色为本人所有」授权")

    _check_audio_file(voice, "干声")
    _check_audio_file(song, "歌曲")
    if JOB_QUEUE.qsize() >= MAX_QUEUE:
        raise HTTPException(409, "当前排队任务较多，请稍后再试")

    job = _new_job(voice.filename or "voice", song.filename or "song", 0.0)
    upload_dir = Path(job["dir"]) / "upload"

    for field, base in ((voice, "voice"), (song, "song")):
        ext = Path(field.filename or "").suffix.lower()
        target = upload_dir / (base + ext)   # 保留扩展名：试听可回原始完整上传件
        size = 0
        with target.open("wb") as out:
            while chunk := field.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    out.close()
                    shutil.rmtree(job["dir"], ignore_errors=True)
                    with _jobs_lock:
                        JOBS.pop(job["job_id"], None)
                    raise HTTPException(413, f"文件过大（上限 500MB）")
                out.write(chunk)
        await field.close()

    # 同步校验：可解码 / 时长 / 音量
    try:
        voice_dur = _ffprobe_duration(_upload_file(Path(job["dir"]), "voice"))
        song_dur = _ffprobe_duration(_upload_file(Path(job["dir"]), "song"))
    except ValueError:
        _drop(job)
        raise HTTPException(400, "音频无法解码，请确认文件未损坏（支持 mp3 / wav / m4a / flac）")

    if voice_dur < 3:
        _drop(job)
        raise HTTPException(400, f"音色样本过短（{voice_dur:.1f} 秒），请提供至少 3 秒、建议 10–25 秒的清唱干声")
    if voice_dur > 300:
        _drop(job)
        raise HTTPException(400, f"音色样本过长（{voice_dur / 60:.0f} 分钟，上限 5 分钟）")
    if song_dur > 720:
        _drop(job)
        raise HTTPException(400, f"歌曲过长（{song_dur / 60:.0f} 分钟），DEMO 建议使用 3–5 分钟歌曲")
    if song_dur < 15:
        _drop(job)
        raise HTTPException(400, "歌曲过短（至少 15 秒）")

    v_db = _volume_db(_upload_file(Path(job["dir"]), "voice"))
    if v_db is not None:
        if v_db <= -70:
            _drop(job)
            raise HTTPException(400, "音色样本全静音，请检查录音")
        if v_db < -45:
            _drop(job)
            raise HTTPException(400, "音色样本音量过低，请提高录音音量后重新上传")
    s_db = _volume_db(_upload_file(Path(job["dir"]), "song"))
    if s_db is not None and s_db <= -60:
        _drop(job)
        raise HTTPException(400, "歌曲文件全静音，请检查文件")

    _set(job, song_duration=round(song_dur, 1))
    JOB_QUEUE.put(job["job_id"])
    return {"job_id": job["job_id"], "batch_id": job["job_id"], "status": "queued"}


def _drop(job: dict) -> None:
    with _jobs_lock:
        JOBS.pop(job["job_id"], None)
    shutil.rmtree(job["dir"], ignore_errors=True)


def _get_job(job_id: str) -> dict:
    with _jobs_lock:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在或已清理")
    return job


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    return _public(_get_job(job_id))


@app.get("/jobs/{job_id}/result")
def get_result(job_id: str, fmt: str = "mp3", kind: str = "mix"):
    """下载成品：kind=mix（默认，成品）| vocal（AI 人声分轨）| bg（伴奏分轨）。
    分轨仅提供 WAV（无损，适合导入 DAW/声卡效果链）。"""
    job = _get_job(job_id)
    if job.get("status") != "done" or not job.get("result"):
        raise HTTPException(409, "任务尚未完成" if job.get("status") == "running" else "任务未成功完成，无法下载")
    fmt = (fmt or "mp3").lower()
    kind = (kind or "mix").lower()
    if fmt not in ("mp3", "wav"):
        raise HTTPException(400, "fmt 仅支持 mp3 或 wav")
    if kind not in ("mix", "vocal", "bg"):
        raise HTTPException(400, "kind 仅支持 mix / vocal / bg")
    job_dir = Path(job["dir"])
    stem = _safe_stem(job["song_name"])
    if kind == "mix":
        if fmt == "wav":
            path, media, name = job_dir / "mix.wav", "audio/wav", "mix.wav"
        else:
            path, media, name = job_dir / f"{stem}_AI.mp3", "audio/mpeg", f"{stem}_AI.mp3"
    else:
        if fmt != "wav":
            raise HTTPException(400, "分轨仅提供 WAV（无损，适合导入 DAW）")
        if kind == "vocal":
            path, name = job_dir / "ai.wav", f"{stem}_AI人声.wav"
        else:
            path, name = job_dir / "bg.wav", f"{stem}_伴奏.wav"
        media = "audio/wav"
    if not path.exists():
        raise HTTPException(404, "成品文件缺失，请重新生成")
    return FileResponse(str(path), media_type=media, filename=name)


@app.get("/jobs/{job_id}/audio/{kind}")
def get_audio(job_id: str, kind: str):
    """对比试听：cover=翻唱 / song=原唱 / voice=本人干声 / vocal=AI 人声分轨 / bg=伴奏分轨。"""
    job = _get_job(job_id)
    if kind not in ("cover", "song", "voice", "vocal", "bg"):
        raise HTTPException(400, "kind 仅支持 cover / song / voice / vocal / bg")
    if job.get("status") != "done":
        raise HTTPException(409, "任务尚未完成，暂无可试听音频")
    job_dir = Path(job["dir"])
    stem = _safe_stem(job["song_name"])
    candidates = {
        "cover": [job_dir / f"{stem}_AI.mp3", job_dir / "mix.wav"],
        # 干声：优先原始完整上传件（保留扩展名，可推断 content-type）；
        # 老任务无扩展名上传 → 回退 voice.wav（截断 15 秒版，仅为兼容）
        "voice": [p for p in [ _upload_file(job_dir, "voice") ] if p and p.suffix]
                or [job_dir / "voice.wav"],
        "song": [p for p in [ _upload_file(job_dir, "song") ] if p and p.suffix]
                or [job_dir / "song.wav"],
        "vocal": [job_dir / "ai.wav"],
        "bg": [job_dir / "bg.wav"],
    }[kind]
    for p in candidates:
        if p.exists():
            media = _AUDIO_MEDIA.get(p.suffix.lower(), "application/octet-stream")
            return FileResponse(str(p), media_type=media)
    raise HTTPException(404, "音频文件已清理")


@app.get("/history")
def history(offset: int = 0, limit: int = 10):
    """翻唱历史（按批次聚合，仅 done）：批次 = 一次生成操作，含干声/源歌/翻唱。
    分页：offset + limit（默认每页 10 批次），返回 total 与 has_more。"""
    offset = max(0, offset)
    limit = max(1, min(limit, 50))
    try:
        total = _db_query("SELECT COUNT(*) AS n FROM jobs WHERE status='done'")[0]["n"]
        rows = _db_query(
            "SELECT * FROM jobs WHERE status='done' ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset))
    except sqlite3.Error:
        total, rows = 0, []
    items = []
    for r in rows:
        job_dir = JOBS_DIR / r["job_id"]
        fn = r["result_filename"] or ""
        items.append({
            "batch_id": r["job_id"],
            "job_id": r["job_id"],
            "song_name": r["song_name"],
            "voice_name": r["voice_name"],
            "created_at": r["created_at"],
            "finished_at": r["finished_at"],
            "filename": fn,
            "duration": r["result_duration"],
            "size_bytes": r["result_size_bytes"],
            "diffusion_steps": r["diffusion_steps"],
            "pitch_shift": r["pitch_shift"],
            "reverb_wet": r["reverb_wet"],
            "stats": {
                "total_s": r["total_s"], "speed_x": r["speed_x"],
                "peak_rss_gb": r["peak_rss_gb"],
                "demucs_peak_rss_gb": r["demucs_peak_rss_gb"],
                "mps_driver_gb": r["mps_driver_gb"],
            },
            "files": {
                "cover": bool(fn) and (job_dir / fn).exists(),
                "voice": (_upload_file(job_dir, "voice") is not None) or (job_dir / "voice.wav").exists(),
                "song": (_upload_file(job_dir, "song") is not None) or (job_dir / "song.wav").exists(),
                "vocal": (job_dir / "ai.wav").exists(),
                "bg": (job_dir / "bg.wav").exists(),
            },
        })
    return {"items": items, "total": total, "offset": offset, "has_more": offset + len(items) < total}


@app.get("/api/config")
def get_config():
    return {
        "config": _get_config(),
        "qualities": [{"id": q, "label": QUALITY_LABEL[q], "steps": QUALITY_STEPS[q]}
                      for q in ("fast", "standard", "fine")],
        "model": {
            "engine": "SEED-VC · 歌声转换 · 44.1kHz（f0）",
            "separator": "Demucs · HTDemucs 人声分离",
            "device": "Apple Silicon MPS · 本机推理",
            "online": False,
        },
    }


@app.post("/api/config")
def save_config(payload: dict):
    cfg = _get_config()
    if "quality" in payload:
        if payload["quality"] not in QUALITY_STEPS:
            raise HTTPException(400, "quality 仅支持 fast / standard / fine")
        cfg["quality"] = payload["quality"]
    if "pitch_shift" in payload:
        try:
            cfg["pitch_shift"] = int(payload["pitch_shift"])
        except (TypeError, ValueError):
            raise HTTPException(400, "pitch_shift 须为 -12 ~ 12 的整数")
        cfg["pitch_shift"] = max(-12, min(12, cfg["pitch_shift"]))
    if "auto_f0" in payload:
        cfg["auto_f0"] = bool(payload["auto_f0"])
    if "reverb_wet" in payload:
        try:
            cfg["reverb_wet"] = round(float(payload["reverb_wet"]), 2)
        except (TypeError, ValueError):
            raise HTTPException(400, "reverb_wet 须为 0 – 0.3 的数字")
        cfg["reverb_wet"] = max(0.0, min(0.3, cfg["reverb_wet"]))
    _save_config(cfg)
    return {"config": _get_config()}


@app.get("/")
def index():
    return FileResponse(str(ROOT / "static" / "index.html"), media_type="text/html")


@app.get("/api/ping")
def ping():
    return {"ok": True, "ffmpeg": bool(shutil.which("ffmpeg"))}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8300)
