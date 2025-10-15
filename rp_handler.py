import base64, io, os, random, tempfile, time
from typing import Any, Dict, Optional

import torch
import torchaudio
import numpy as np

import runpod
from runpod.serverless.utils.rp_download import file as rp_file
from runpod.serverless.modules.rp_logger import RunPodLogger

# --- актуальные импорты из DiffRhythm ---
from infer.infer import inference
from infer.infer_utils import (
    get_lrc_token,
    get_negative_style_prompt,
    get_reference_latent,
    get_style_prompt,
    prepare_model,
)

LOGGER = RunPodLogger()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
SR = 44100

# Кэшим модели по длине (2048 | 6144)
_CACHE = {}  # key=max_frames -> dict(models...)

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("TORCH_HOME", "/workspace/.cache/torch")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _download_to_local(url: str) -> Optional[str]:
    try:
        info = rp_file(url)
        return info["file_path"]
    except Exception as e:
        LOGGER.error(f"Download failed: {e}")
        return None


def _is_valid_duration(x: int) -> bool:
    return x == 95 or (96 <= x <= 285)


def _wav_b64_from_numpy(x: np.ndarray, sr: int = SR) -> str:
    """x shape: (samples, channels) or (channels, samples) 
    — приведём к (channels, samples)."""
    if x.ndim == 2 and x.shape[0] < x.shape[1]:
        # (channels, samples) ok
        pass
    elif x.ndim == 2:
        x = x.T
    elif x.ndim == 1:
        x = np.expand_dims(x, 0)

    tensor = torch.from_numpy(x.astype(np.float32))
    buf = io.BytesIO()
    torchaudio.save(buf, tensor, sample_rate=sr, format="wav")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _wav_b64_from_tensor(t: torch.Tensor, sr: int = SR) -> str:
    buf = io.BytesIO()
    torchaudio.save(buf, t.cpu(), sample_rate=sr, format="wav")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _load_models(max_frames: int,
                 device: str):
    if max_frames not in _CACHE:
        LOGGER.info(f"Loading models for max_frames={max_frames} on {device}")
        cfm, tokenizer, muq, vae = prepare_model(max_frames, device)
        _CACHE[max_frames] = {"cfm": cfm,
                              "tokenizer": tokenizer,
                              "muq": muq, "vae": vae}
    m = _CACHE[max_frames]
    return m["cfm"], m["tokenizer"], m["muq"], m["vae"]


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    input:
      lyric: str (LRC формат можно пустым)
      audio_length: 95|285
      ref_prompt: str  ИЛИ  ref_audio_url: str
      repo_id: str (опционально: ASLP-lab/DiffRhythm-1_2, -full и т.д.)
      chunked: bool (по умолчанию True)
      seed: int (опционально)
    """
    t0 = time.time()
    inp = job.get("input", {}) or {}

    lyric = inp.get("lyric", "")
    audio_length = int(inp.get("audio_length", 95))
    music_duration = int(inp.get("music_duration", audio_length))
    if not _is_valid_duration(audio_length):
        return {"error": "audio_length must be 95 or between 96 and 285."}
    if not _is_valid_duration(music_duration):
        return {"error": "music_duration must be 95 or between 96 and 285."}
    max_frames = 2048 if audio_length == 95 else 6144

    ref_prompt = inp.get("ref_prompt", "")
    ref_audio_url = inp.get("ref_audio_url", "")
    if not (ref_prompt or ref_audio_url):
        return {"error": "Either 'ref_prompt' or 'ref_audio_url' must be provided."}  # noqa
    if ref_prompt and ref_audio_url:
        return {"error": "Use only one: 'ref_prompt' OR 'ref_audio_url'."}

    chunked = bool(inp.get("chunked", True))
    seed = int(inp.get("seed", random.randint(0, 2**31 - 1)))
    torch.manual_seed(seed)

    try:
        cfm, tokenizer, muq, vae = _load_models(max_frames, DEVICE)

        # токены лирики (актуальная сигнатура)
        lrc_prompt, start_time, end_frame, song_duration = get_lrc_token(
            max_frames, lyric, tokenizer, music_duration, DEVICE
        )

        # стиль из текста или аудио
        if ref_prompt:
            style_prompt = get_style_prompt(muq, prompt=ref_prompt)
        else:
            local = _download_to_local(ref_audio_url)
            if not local:
                return {"error": "Failed to download 'ref_audio_url'."}
            style_prompt = get_style_prompt(muq, local)

        negative_style_prompt = get_negative_style_prompt(DEVICE)

        # латент + предсегменты (без редактирования)
        latent_prompt, pred_frames = get_reference_latent(DEVICE,
                                                          max_frames,
                                                          False,
                                                          None,
                                                          None,
                                                          vae)

        # минимально корректный вызов inference,
        #   как в их CLI (без generator и без eval_song)
        with torch.autocast(device_type="cuda",
                            enabled=(DEVICE == "cuda"),
                            dtype=DTYPE):
            out = inference(
                cfm_model=cfm,
                vae_model=vae,
                cond=latent_prompt,
                text=lrc_prompt,
                duration=end_frame,
                style_prompt=style_prompt,
                negative_style_prompt=negative_style_prompt,
                start_time=start_time,
                pred_frames=pred_frames,
                batch_infer_num=1,
                song_duration=song_duration,
                chunked=chunked
            )

        # out — это список int16-тензоров формы (channels, samples)
        if not isinstance(out, list) or len(out) == 0:
            return {"error": "Inference returned empty output."}

        audio_i16 = out[0]
        if audio_i16.ndim == 1:
            audio_i16 = audio_i16[None, :]
        elif audio_i16.ndim == 2 and audio_i16.shape[0] > audio_i16.shape[1]:
            audio_i16 = audio_i16.T

        tmp = tempfile.mkdtemp()
        out_wav = os.path.join(tmp, "output.wav")
        torchaudio.save(out_wav, audio_i16.cpu(), sample_rate=SR)

        # base64, если нужно вернуть в ответе:
        b64 = _wav_b64_from_tensor(audio_i16, SR)
        return {
            "output_path": out_wav,
            "audio_base64": b64,
            "sample_rate": SR,
            "audio_length": audio_length,
            "seed": seed,
            "chunked": chunked,
            "time_sec": round(time.time() - t0, 3),
        }

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        msg = str(e)
        if "CUDA out of memory" in msg:
            return {"error": "CUDA OOM — попробуйте audio_length=95 и оставьте chunked=True.",  # noqa
                    "detail": msg}
        return {"error": msg}
    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc(limit=8)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
