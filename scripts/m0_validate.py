"""
M0 — validate the local GPU speech-to-text stack on this machine.

Purpose: prove that faster-whisper large-v3 actually runs on the RTX 5080
(Blackwell / sm_120) using CUDA — NOT silently on CPU — and is faster than
real time. This de-risks the single most fragile assumption in the whole
project before any app code is written.

Exit code 0 = GPU STT works. Non-zero = it does not; read the diagnostics and
fall back per DESIGN.md §6 (whisper-rs+CUDA, or openai-whisper on torch cu128).
"""
import sys
import time
import wave
import contextlib
from pathlib import Path

# This machine's console is cp932 (Japanese Windows); force UTF-8 so non-ASCII
# output (em-dashes, etc.) can't crash the run.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

MODEL = "large-v3"
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WAV = ROOT / "models" / "test.wav"


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def wav_duration_seconds(path):
    with contextlib.closing(wave.open(str(path), "rb")) as w:
        return w.getnframes() / float(w.getframerate())


def check_torch_gpu():
    """torch is our canary: its cu128 wheels have sm_120 kernels, so a working
    torch.cuda confirms the GPU itself is usable from Python even if ctranslate2
    (faster-whisper's backend) turns out not to support Blackwell yet."""
    section("1. GPU sanity check (torch)")
    try:
        import torch
    except ImportError:
        print("  torch not installed — skipping (faster-whisper uses ctranslate2, "
              "not torch, so this is only an extra canary).")
        return None
    print(f"  torch version      : {torch.__version__}")
    avail = torch.cuda.is_available()
    print(f"  cuda available     : {avail}")
    if not avail:
        print("  !! torch cannot see CUDA. Driver / wheel mismatch likely.")
        return False
    cap = torch.cuda.get_device_capability(0)
    print(f"  device             : {torch.cuda.get_device_name(0)}")
    print(f"  compute capability : sm_{cap[0]}{cap[1]} (Blackwell = sm_120)")
    print(f"  cuda (torch build) : {torch.version.cuda}")
    # Tiny real op to confirm kernels actually launch on this arch.
    try:
        x = torch.randn(2048, 2048, device="cuda")
        torch.mm(x, x)
        torch.cuda.synchronize()
        print("  kernel launch test : OK")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  !! kernel launch FAILED: {e}")
        print("     -> sm_120 kernels missing from this torch build.")
        return False


def run_faster_whisper(wav_path):
    section("2. faster-whisper large-v3 on CUDA")
    try:
        from faster_whisper import WhisperModel
    except Exception as e:  # noqa: BLE001 — show the REAL error, don't assume "not installed"
        print(f"  !! could not import faster_whisper: {type(e).__name__}: {e}")
        print("     (if this is a missing module, add it to requirements.txt and "
              "re-run Setup-M0.ps1)")
        return False

    print(f"  loading {MODEL} (device=cuda, compute_type=float16)...")
    t0 = time.perf_counter()
    try:
        model = WhisperModel(MODEL, device="cuda", compute_type="float16")
    except Exception as e:  # noqa: BLE001
        print(f"  !! model load on CUDA FAILED: {e}")
        print("     ctranslate2 likely lacks sm_120 support. See DESIGN.md §6 "
              "fallback (whisper-rs+CUDA, or openai-whisper on torch cu128).")
        return False
    load_s = time.perf_counter() - t0
    print(f"  model loaded in    : {load_s:.1f}s")

    audio_s = wav_duration_seconds(wav_path)
    print(f"  test audio         : {wav_path.name} ({audio_s:.1f}s)")

    # Warm-up pass (first inference includes CUDA graph / kernel JIT cost).
    list(model.transcribe(str(wav_path), beam_size=1)[0])

    t0 = time.perf_counter()
    segments, info = model.transcribe(str(wav_path), beam_size=5, language="en")
    text = " ".join(s.text.strip() for s in segments)
    proc_s = time.perf_counter() - t0

    rtf = proc_s / audio_s if audio_s else float("inf")
    print(f"  processing time    : {proc_s:.2f}s")
    print(f"  real-time factor   : {rtf:.2f}x  ({'FASTER' if rtf < 1 else 'SLOWER'} than real time)")
    print(f"  detected language  : {info.language} (p={info.language_probability:.2f})")
    print(f"\n  TRANSCRIPT:\n  {text}")

    truth_path = wav_path.with_suffix(".txt")
    if truth_path.exists():
        truth = truth_path.read_text(encoding="utf-8").strip()
        wer = simple_wer(truth, text)
        print(f"\n  ground truth:\n  {truth}")
        print(f"\n  word error rate    : {wer:.1%}")

    return rtf < 1.5  # generous bar; we expect << 1.0 on a 5080


def simple_wer(ref, hyp):
    """Levenshtein word error rate — rough accuracy signal, not production-grade."""
    import re
    r = re.findall(r"[a-z']+", ref.lower())
    h = re.findall(r"[a-z']+", hyp.lower())
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(r)][len(h)] / max(len(r), 1)


def main():
    wav_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WAV
    if not wav_path.exists():
        print(f"Test WAV not found: {wav_path}\nRun scripts\\Make-TestWav.ps1 first.")
        return 2

    torch_ok = check_torch_gpu()
    stt_ok = run_faster_whisper(wav_path)

    section("VERDICT")
    if stt_ok:
        print("  PASS — faster-whisper large-v3 runs on the GPU, faster than real")
        print("  time. M0 cleared; proceed to M1 (dual-track capture).")
        return 0
    print("  FAIL — GPU STT did not work as required.")
    if torch_ok:
        print("  torch sees the GPU fine, so the GPU is usable from Python — the")
        print("  problem is ctranslate2/Blackwell. Try the openai-whisper fallback")
        print("  (runs on torch cu128) or whisper-rs+CUDA per DESIGN.md §6.")
    else:
        print("  torch also could not use the GPU — suspect driver / CUDA wheel")
        print("  mismatch before blaming faster-whisper.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
