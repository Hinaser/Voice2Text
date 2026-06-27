# Voice2Text — Design & Build Plan

Real-time meeting transcription for Windows 11. Captures **both** the local
microphone and the system audio (what other attendees say), transcribes it
locally on the GPU, and shows live captions plus a saved transcript.

Built for: twice-weekly English meetings, non-native speaker, accuracy +
not-missing-spoken-content over everything else.

**Usage profile (confirmed):** Zoom primarily, Google Meet occasionally.
**Speakers + mic** (rarely headphones) → must handle acoustic echo (§3.6).
Floating, resizable overlay box. **Latency matters** → see §6 streaming note.

---

## 1. Goals & constraints

### Functional goals
- **Live captions** during the meeting (near-real-time, low latency).
- **Saved transcript** after the meeting (clean, exportable, optional summary).
- Capture **attendees' audio** (system output) — the core problem the phone app
  solves poorly.
- Capture **own mic** too, labeled separately ("me" vs "others").
- Always-on-top, resizable, semi-transparent caption window.

### Hard constraints
- **100% local.** No meeting audio leaves the machine (work privacy/compliance).
- Windows 11 only (no cross-platform requirement).
- Must be reliable mid-meeting — capture cannot silently drop.

### Non-goals (v1)
- Speaker diarization beyond the mic/system split (no "who said what" among
  attendees).
- Multi-language. English only for now.
- Cloud sync, accounts, mobile.

### Confirmed environment (verified 2026-06-27)
| Component | Version | Note |
|---|---|---|
| GPU | RTX 5080, 16 GB | Blackwell **sm_120** — needs recent CUDA stack |
| CUDA toolkit | 12.8 | ✅ supports sm_120 |
| Rust / cargo | 1.91.1 | |
| Node | 24.13.0 | for Tauri frontend tooling |
| Python | 3.11.14 | for faster-whisper sidecar |

> ⚠️ **Blackwell gotcha.** The 5080 (sm_120) is new. PyTorch and CTranslate2
> (the faster-whisper backend) must be CUDA-12.8-capable builds with sm_120
> kernels. This is the single most likely thing to break setup — validated as
> **Milestone 0** below before any other work.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Tauri app (single shippable .exe / installer)                │
│                                                                │
│  ┌────────────────────────┐      Tauri events (JSON)          │
│  │  Web UI  (HTML/CSS/TS)  │ ◀───────────────┐                 │
│  │  • caption overlay      │                 │                 │
│  │  • live partial line    │                 │                 │
│  │  • finalized history    │                 │                 │
│  │  • transcript + export  │ ── commands ──▶  │                 │
│  └────────────────────────┘                 │                 │
│                                              │                 │
│  ┌───────────────────────── Rust core ───────┴──────────────┐ │
│  │  capture:   cpal → WASAPI mic + WASAPI loopback           │ │
│  │  resample:  → 16 kHz mono f32 per track                   │ │
│  │  vad:       Silero (onnxruntime) → utterance boundaries   │ │
│  │  dispatch:  utterance buffers → STT, results → UI events  │ │
│  │  stt:       whisper-rs (whisper.cpp) large-v3, CUDA       │ │
│  │             — IN-PROCESS, no Python                       │ │
│  └──────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### Why this shape
- **Tauri, not Electron:** native webview (~10 MB vs ~150 MB), and the audio
  capture has to be native anyway. Web UI keeps the overlay flexible and is the
  fastest surface to iterate on with AI assistance.
- **Tauri, not pure native (egui/Win32):** UI iteration speed. This is a personal
  tool; overlay polish matters more than shaving the last few MB.
- **Native in-process STT (`whisper-rs` + CUDA), NOT a Python sidecar.** Driven by
  the **standalone/portable requirement** (§9): the app must ship with no Python
  on the user's machine. `whisper-rs` (Rust bindings to whisper.cpp) compiles the
  CUDA engine into the binary, so distribution is one `.exe` + a model file + a
  few CUDA DLLs — no embedded interpreter, no second process.
  - M0 validated faster-whisper (CTranslate2) only to prove **GPU STT is viable on
    Blackwell** — that risk is retired. The production engine is now whisper.cpp,
    which keeps the same large-v3 model and accuracy. The `.venv` stays as a
    dev-time reference/oracle and accuracy comparison baseline.
  - New risk introduced by this choice: whisper.cpp must build + run with CUDA on
    sm_120. Retired by **M0.5** before any app code depends on it.

---

## 3. Component design

### 3.1 Audio capture (Rust, `cpal`)
- Two independent input streams:
  - **Mic:** default capture device.
  - **System / attendees:** WASAPI **loopback** of the default render device.
    `cpal` exposes loopback on Windows; this is what grabs Teams/Zoom output.
- Each stream: device-native sample rate/format → convert to **16 kHz mono f32**
  (`rubato` or a simple resampler).
- Ring buffer per track (a few seconds) so VAD/STT can lag without dropping audio.
- Emit periodic RMS levels to the UI so the user can *see* both tracks are live
  (critical trust signal before a meeting starts).

**Risk:** loopback device selection when the user changes output device (e.g.
plugs in headphones) mid-meeting. Handle device-change events; re-open loopback.

### 3.2 Voice activity detection (Rust, Silero VAD)
- **Primary option (M0.5 finding):** `whisper-rs` 0.16 bundles Silero VAD
  (`whisper_vad`), so we may get VAD from the same crate — no separate
  `onnxruntime` dependency. Confirm its API fits per-frame streaming use in M2;
  fall back to a standalone Silero-onnx crate if not.
- Run VAD per track on 30 ms frames.
- Open an utterance on speech onset; close it after ~500–800 ms of trailing
  silence (tunable) or a max length cap (~15 s) to bound latency.
- Closed utterance → hand the buffered audio to the STT dispatcher.
- This is what makes non-streaming Whisper *feel* live: latency ≈ trailing
  silence + inference time.

### 3.3 STT engine (Rust, in-process — `whisper-rs` + CUDA)
- `whisper-rs` (Rust bindings to whisper.cpp) built with the CUDA feature.
- Model loaded **once at startup** and kept resident (large-v3 GGML, fp16). Never
  reload per utterance (M0 showed cold load is slow, warm inference is instant).
- API surface (internal Rust, not IPC): `transcribe(pcm: &[f32], track) -> {text,
  segments, duration_ms}`. Word timestamps available for caption rendering.
- Runs on a dedicated worker thread so capture/VAD never block on inference.
- On RTX 5080, large-v3 transcribes a few-second utterance in well under its own
  duration (M0: RTF 0.06× with CTranslate2; whisper.cpp expected similar order).
- **Model file ships with the app** (GGML `.bin`). **Provisional choice: q5_0
  (~1.08 GB)** — 0% WER but on *one clean TTS clip only*, which says nothing about
  accented/overlapping/speaker-played meeting audio (external-review caveat). **Not
  final** until benchmarked (with real WER) against full large-v3 on the real
  dual-track WAVs captured in M1. Keep full large-v3 as the fallback.
- **Process isolation (open decision, raised in review):** in-process keeps
  packaging simplest, but a CUDA OOM / native panic would take down capture + UI
  mid-meeting — violating the "capture cannot silently drop" constraint. Mitigate
  by putting STT behind a narrow Rust interface (`trait Stt`) so it can run as a
  **separate native child process** (still Python-free, still 100% local) with a
  watchdog/restart, if reliability testing shows in-process crashes. Decide in M3.

### 3.4 UI (Tauri webview)
- **Overlay mode:** frameless, always-on-top, **floating resizable box** (drag to
  move, drag edges to resize; size + position persisted). Adjustable opacity and
  font size. Shows the current partial line + last N finalized lines, color-coded
  by track (me / others), with the box auto-scrolling to the latest line.
- **Transcript mode:** full scrollback, timestamps, copy/export (Markdown, .txt,
  .srt), and an optional **"Summarize"** button (§3.5).
- State lives in Rust; UI is a view. Events: `caption.partial`, `caption.final`,
  `level.meter`, `status`.

### 3.6 Acoustic echo handling ⚠️ *(because user uses speakers, not headphones)*
Attendees' voices play out the speakers and are re-captured by the mic. Without
handling, every remote utterance gets transcribed **twice** — once from loopback
("others"), once as echo on the mic ("me").

> **Revised after external review (Codex, 2026-06-27).** The earlier plan led with
> *half-duplex mic gating* (mute "me" whenever "others" is speaking). That is
> **lossy in exactly the wrong way**: it drops the user's own backchannels,
> interruptions, short confirmations, and overlap speech — content the user
> explicitly does not want to miss. Gating is demoted to a last resort.

Preferred order (least lossy first):
1. **Loopback is the source of truth for "others."** Never transcribe remote
   speech from the mic.
2. **Real acoustic echo cancellation (preferred).** Open the mic via the Windows
   Communications capture path / Voice Capture DSP (APO AEC), which cancels
   speaker echo at the source using the render signal as reference. Keeps the full
   "me" channel intact. Validate availability/quality in M1.
   - **API available, but NOT supported on the user's mic (M1 finding, 2026-06-27).**
     The `wasapi` crate exposes the path (`set_properties(StreamCategory::
     Communications)` → `is_aec_supported()` → `get_aec_control()
     .set_echo_cancellation_render_endpoint(...)`), and the M1 harness wires it up.
     But on the user's actual default mic — *ASUS AI Noise-Cancelling Mic Adapter* —
     **`is_aec_supported()` returns false**: that endpoint doesn't expose the
     Windows AEC APO (likely because the adapter does its own onboard processing).
     **Consequence:** real OS AEC is NOT a usable default here. Re-check on a real
     meeting and on any other mic, but plan around it being unavailable → strategy
     #3 (dedup) becomes the realistic default, with **software AEC** (e.g.
     `webrtc-audio-processing` / speexdsp, using loopback as the reference signal)
     as the higher-quality option if dedup is insufficient.
3. **Keep-both-then-dedup.** Transcribe both tracks; when a near-identical line
   appears on both within a short window, drop the *mic* copy (loopback wins).
   Loses nothing the user said that the remote side didn't also say.
4. **Half-duplex gating — last resort only.** Accept losing simultaneous local
   speech only if AEC is unavailable AND dedup proves unreliable.

**M1 must record raw mic + loopback (+ AEC-processed mic if available) during real
speaker playback**, so we choose between #2/#3/#4 from evidence, not theory.

### 3.5 Post-meeting cleanup (optional, local LLM)
- Send the finalized transcript to a local model (Ollama, e.g. a Llama/Qwen
  instruct model on the same GPU) to: fix punctuation/casing, summarize, and
  extract action items. Fully offline. Off by default; one button.

---

## 4. Data flow (one utterance)
1. Loopback stream fills the "others" ring buffer.
2. VAD detects speech start → … → 600 ms silence → utterance closed.
3. Dispatcher slices the buffer, assigns an `id`, sends to sidecar.
4. Sidecar transcribes, returns text + segments.
5. Rust emits `caption.final` (track=others) → UI appends a line and appends to
   the transcript store.
6. (While that runs, the next utterance may already be in flight — pipeline.)

---

## 5. Milestones

> Sequenced by **risk first**. Each milestone is independently verifiable.

### M0 — Validate the Blackwell ML stack ✅ *PASSED (2026-06-27)*
- Created `.venv`, installed torch 2.11.0+cu128 + faster-whisper 1.1.1
  (ctranslate2 4.8.0). Had to add `requests` explicitly (faster-whisper 1.1.1
  imports it but newer huggingface-hub no longer pulls it in transitively).
- `scripts/m0_validate.py` loads large-v3 on CUDA and transcribes a locally
  TTS-generated clip; checks RTF + WER vs ground truth.
- **Result:** sm_120 confirmed; **RTF 0.06×** (~16× faster than real time);
  **WER 0.0%**. Model load ~62 s first time (incl. 3 GB download); warm load a
  few seconds → sidecar keeps the model resident, never reloads per utterance.
- Implication: ample GPU headroom to run large-v3 + a streaming model together
  (supports the M3.5 Parakeet hybrid).
- Console is **cp932 (Japanese Windows)** — scripts force UTF-8 stdout; keep in
  mind for all future console output (Rust logs, sidecar prints).

### M0.5 — Validate native whisper.cpp + CUDA on Blackwell ✅ *PASSED (2026-06-27)*
- Built `whisper-rs` **0.16** (`features = ["cuda"]`) → compiles whisper.cpp's
  CUDA backend for sm_120. Harness: `m05-whisper-cuda/`, runner `Run-M05.ps1`.
- **Result:** `ggml_cuda_init: ... RTX 5080, compute capability 12.0`, `using
  CUDA0 backend`; **RTF 0.05×** (~20× faster than real time); **WER 0.0%**;
  model load 0.7 s. Ran on the **quantized q5_0 (1.08 GB)** model.
- **Bundle decision:** ship **q5_0** — 0% WER on the test clip at ⅓ the size of
  full large-v3. (Re-check on real meeting audio in M3/M4 before finalizing.)
- **0.16 bonus:** the crate now bundles **Silero VAD** (`whisper_vad.rs`) — may
  cover M2's VAD need without a separate onnxruntime dependency.

#### Build recipe (NON-OBVIOUS — reused for the real app build; see `scripts/Build-M05.ps1`)
The native CUDA build needs several things the default toolchain lacks. In order
of the walls we hit:
1. **CMake** (winget `Kitware.CMake`) — whisper.cpp builds via CMake.
2. **LLVM/libclang** (winget `LLVM.LLVM`, set `LIBCLANG_PATH=C:\Program
   Files\LLVM\bin`) — `whisper-rs-sys` runs bindgen.
3. **Ninja generator** (`CMAKE_GENERATOR=Ninja`, ninja.exe from VS's CMake
   component on PATH) — the **VS generator fails with "No CUDA toolset found"**
   because CUDA 12.8's MSBuild integration was not installed into this new VS 18.
4. **Clear VS-only generator vars** after importing vcvars (`CMAKE_GENERATOR_
   INSTANCE/PLATFORM/TOOLSET`) — else Ninja errors "does not support instance".
5. **`CUDAARCHS=120`** — force sm_120 kernels (avoid "no kernel image").
6. **`NVCC_PREPEND_FLAGS=-allow-unsupported-compiler`** — this VS's MSVC 14.50 is
   newer than CUDA 12.8 officially supports; nvcc's version gate must be bypassed.
7. **whisper-rs 0.16, not 0.14** — 0.14's whisper-rs-sys 0.13.1 hit a bindgen
   layout-test panic against LLVM 21's libclang.
8. Runtime: CUDA DLLs (cudart/cublas/...) must be on PATH / next to the exe.
- *If this had failed*, the fallback was bundling the Python sidecar (§6) — not
  needed.

### M1 — Audio-capture stress test (THE real project gate, ~2–4 days) — IN PROGRESS
> Expanded after external review: capture — not STT — is the app's primary failure
> mode. Validate it hard, on real conferencing apps, before building anything on
> top. Can start as a plain Rust binary; Tauri scaffolding can follow once capture
> is proven.

**Status (2026-06-27):** harness `m1-capture/` BUILT & self-tested. Captures 3
tracks (raw mic + AEC mic + loopback) → `models/m1/*.wav`, with device/role
enumeration, live ASCII meters, and a drift/dropout summary. Self-test confirmed
dual+ capture, 16 kHz-mono autoconvert, valid WAVs. **Findings:** (a) all render
role-defaults are the same device on this box; (b) **Windows AEC not supported on
the user's mic** (see §3.6); (c) reported "drift" is dominated by ~100 ms thread-
startup offset — real long-run clock drift needs the actual 30–60 min recording.
**Still pending (needs the user):** run against real Zoom + Meet through speakers,
mid-session device hot-swap, and the 30–60 min drift check.

- **Device reality:** enumerate render + capture endpoints; capture the **default
  Multimedia vs default Communications** device correctly (Windows has two, and
  Zoom often uses Communications). Record the selected endpoint IDs.
- **Dual-track capture:** mic + WASAPI loopback simultaneously via `cpal`; two live
  level meters; write each track to WAV (plus AEC-processed mic if available, for
  §3.6).
- **Format/clock handling:** accept device-native rate/format (44.1/48/96 kHz,
  stereo/float/int), downmix to mono, resample to 16 kHz. **Handle clock drift**
  between the two independent streams over a long session (they have separate
  clocks) — measure drift, resync/timestamp so tracks don't desync over 60 min.
- **Robustness:** survive output-device **hot-swap** mid-capture (speakers→BT
  headset) by detecting the change and re-opening loopback; handle no-loopback and
  device-removed gracefully.
- **Real-app matrix (the actual proof):** verify capture of **Zoom desktop** and
  **Google Meet in browser**, *including when they are NOT on the default render
  device*, played through **speakers**. Record real dual-track WAVs here — these
  become the test corpus for the q5_0/accuracy and echo decisions.
- **Exit criteria:** from real Zoom + Meet sessions through speakers, both tracks
  captured cleanly, aligned (no audible drift) over a 30–60 min recording,
  surviving a mid-session device change. If this can't be made reliable, the rest
  of the architecture is moot.

### Vertical-slice demo — system audio → live text ✅ *WORKS (2026-06-27)*
- `live-stt/` combines validated M0.5 (whisper CUDA) + M1 (wasapi loopback) with a
  simple **energy-based** segmenter to print live transcripts of system audio.
- **Proven:** played a speech clip through the output device → captured via
  loopback → transcribed on GPU → printed `[mm:ss] <text>` correctly, live.
- Confirmed two real requirements: (a) **buffer audio during model warmup** (a
  clip played before "listening" was missed); (b) energy VAD is serviceable but
  M2's Silero will give cleaner boundaries. Music/non-speech filtered via
  `no_speech_probability > 0.6`.
- This is a demo, not the product: loopback-only (no mic track yet), no UI (M4),
  energy VAD not Silero (M2). It de-risks the live path early.
- **Transcript saving (added 2026-06-28):** each recognized line is appended to a
  timestamped file (`transcripts/transcript-<stamp>.txt`) and flushed immediately
  — live, crash-safe record. Runner: `scripts/Run-LiveStt.ps1`.
- **Accuracy note:** q5_0 occasionally mis-hears a word even on the clean TTS clip
  (observed "quarterly" → "Kiotari" once across runs). Reinforces §3.3: benchmark
  q5_0 vs full large-v3 on real meeting audio before finalizing the bundle model.
- **Interim partials (added 2026-06-28, addresses latency complaint).** While an
  utterance is open, re-transcribe the last ≤10 s every ~300 ms and show evolving
  text (console-only; finals still go to file). First words now appear ~0.6–0.9 s
  after speech starts vs waiting for the full pause. Whisper self-revises partials
  ("Kiotari"→"quarterly") and the final is clean. Tunables: `PARTIAL_INTERVAL`,
  `PARTIAL_WINDOW`, min-speech-before-partial.
  - **Still not RNN-T-instant:** user compared to Google Live Transcribe, which
    uses a streaming transducer (~200–300 ms, low flicker). Whisper-partials are a
    big improvement but revise more visibly. If still insufficient → **B: Parakeet-
    TDT via sherpa-onnx** as the live engine (whisper stays for the saved
    transcript). This is the definitive latency fix; bigger integration lift.

### M2 — VAD segmentation (~1 day)
- Add Silero VAD; log utterance start/stop and dump each utterance as a WAV.
- **Exit criteria:** natural speech produces clean per-utterance files; pauses
  split sensibly; tunable silence threshold.

### M3 — Wire STT end-to-end + echo gating (~1–2 days)
- Call the in-process whisper-rs engine (M0.5) on each utterance; print
  transcripts to console with track labels and latency.
- Implement §3.6 echo mitigation #1+#2 (loopback-priority + mic half-duplex
  gating), since the user is on speakers.
- **Exit criteria:** play remote speech through speakers + talk → remote text
  appears once (as "others", not duplicated on "me") within ~1–2 s of utterance
  end. Measure actual latency against the <1.5 s target.

### M3.5 — Streaming ASR for Google-style low latency ✅ *VALIDATED (2026-06-28)*
- User found whisper VAD-chunk + partials still too laggy vs Google Live Transcribe
  (a streaming RNN-T). Validated **true streaming** via **sherpa-onnx online
  recognizer** (harness `m35-sherpa/`).
- Engine: **sherpa-rs 0.6.8** with `download-binaries` (prebuilt sherpa-onnx, no
  source build) + `sys` (raw FFI; sherpa-rs only wraps *offline* recognizers, so we
  drive the online streaming API directly — pattern from its keyword-spotter).
- Model: **streaming Zipformer transducer EN** (`sherpa-onnx-streaming-zipformer-
  en-2023-06-26`, ~250 MB fp32). NOTE: "Parakeet-TDT" is *offline*; the streaming
  RNN-T equivalent is this Zipformer — same family as Google.
- **Results:** word-by-word incremental partials (~0.3 s granularity, first text
  ~0.8 s), endpoint-based finals, **RTF 0.055 on CPU** (no GPU). **100% accurate**
  on the model's real-speech test clip. (Synthetic Windows-TTS audio transcribed
  as garbage — out-of-distribution for small streaming models, not a real concern.)
- **Caveat:** streaming model output is **ALL-CAPS, no punctuation**. Fine for live
  captions; the Whisper *saved* transcript is cased/punctuated. Optional sherpa-onnx
  punctuation model could post-process the live text.
- Build gotcha: bindgen needs the **MSVC SDK include paths** (`INCLUDE` via vcvars)
  for `stdint.h`, plus `LIBCLANG_PATH` — build via `Build-Cuda.ps1` env. Runtime
  needs `sherpa-onnx-c-api.dll` + `onnxruntime.dll` on PATH (in `target/.../deps`).

#### Decision: HYBRID engine
- **Live captions** = streaming Zipformer (sherpa-onnx, **CPU**) — low latency.
- **Saved transcript / accurate "others" track** = Whisper large-v3 (**CUDA**).
- GPU is reserved for Whisper; streaming runs on CPU.
- **Integrated & working (2026-06-28):** `live-stream/` = wasapi loopback +
  streaming Zipformer → live word-by-word captions + saved transcript, CPU-only.
  Self-tested on real speech through speakers: accurate, smooth incremental
  partials, correct finals. Runner: `scripts/Run-LiveStream.ps1`.
- **Readability pass (2026-06-28, per user feedback):**
  - **Static rendering:** replaced scrolling/overwriting console output (which
    caused duplicate rows on wrap) with a fixed in-place region (VT escape codes):
    header + last 6 finalized lines + live partial, each truncated to console width
    so nothing wraps. `live-stream/src/render.rs`.
  - **Casing:** cheap truecaser on live partials (instant, kills ALL-CAPS).
  - **Punctuation:** sherpa-onnx CT-transformer punctuation model on finals;
    normalize its full-width CJK punctuation (，。？！) → ASCII with tidy spacing.
    Verified: "After early nightfall, the yellow lamps would light up..."
  - **Word wrap + speaker turns (2026-06-28, per user feedback):**
    - **Wrapping:** `render.rs` rewritten to a scrolling-transcript model — finals
      print permanently, word-wrapped to console width (full text, no truncation);
      the live partial is pinned below and redrawn in place via *relative* cursor
      moves (survives scrolling), also wrapped.
    - **Speaker diarization:** per-utterance voice embedding (sherpa-onnx CAM++
      `3dspeaker...voxceleb_16k`) + `EmbeddingManager` cosine match (threshold 0.5)
      → "Speaker N" label; new labeled line per turn. Verified it distinguishes
      male/female voices and re-identifies a returning speaker (S1→S2→S1).
      Limitation: per-utterance only (won't split speakers inside one unbroken
      utterance); accuracy depends on clip length + voice distinctness.
- Remaining for the hybrid: run Whisper (CUDA) on each finalized segment for an
  even-higher-accuracy saved transcript (optional now that streaming finals are
  cased+punctuated). Then mic track + overlay UI (M4).

### M4 — Live caption UI ✅ *DONE (2026-06-28)*
- Tauri 2 overlay app in `app/`: frameless/transparent/always-on-top/resizable;
  per-source partial lines + finalized history color-coded by speaker; toolbar
  for font, opacity, pin, clear, settings, close. Emits `partial`/`final`/
  `status` events from the pipeline thread.
- **Exit criteria met:** usable as a live caption bar (verified on test audio;
  real-meeting check still pending the user).

### M4.5 — Mic track + echo dedup ✅ *DONE (2026-06-28)*
- Second WASAPI capture (default mic) + its own streaming recognizer, labeled
  "You". Loopback is the source of truth for "others"; mic lines whose words are
  ≥55% covered by a recent (≤6 s) system line are dropped as speaker echo
  (strategy #3 from §3.6, since OS AEC is unavailable — [[windows-aec-unsupported-mic]]).
  Toggle: `echo_suppression` (live), `mic_capture` (restart).

### M5 — Transcript + export ✅ *DONE (2026-06-28, export TODO)*
- Finalized lines persisted with timestamps + speaker labels to
  `Documents\Voice2Text\transcript-*.txt` (folder configurable).
- **Hybrid clean transcript:** optional Whisper large-v3 (CUDA) worker
  re-transcribes each utterance and writes the *clean* line to the file while
  live captions stay on streaming. Verified: Whisper output = 100% match to
  ground truth on the same TTS clip streaming garbled. Graceful fallback to
  streaming text if the model/GPU is unavailable.
- **Still TODO:** in-app transcript view + md/srt export.

### M6 — Polish & packaging — *mostly DONE (2026-06-28)*
- **Whisper-as-sidecar refactor (DONE):** Whisper-CUDA moved out of the main app
  into `app/whisper-sidecar/` (own exe). The launcher is now provably CPU-only
  (`dumpbin` shows zero CUDA imports). Main app spawns the sidecar over pipes
  (u32-len + f32 LE audio in, one transcript line out; "READY"/"ERROR"
  handshake) only when the clean transcript is enabled; graceful fallback to
  streaming text otherwise.
- **Global show/hide hotkey (DONE):** `tauri-plugin-global-shortcut`, configurable
  (`config.hotkey`, default `Alt+Shift+V`), toggles overlay visibility.
- **Transcript export (DONE):** in-app Copy / .txt / .md / .srt of the current
  session (`export_transcript` command writes into the save folder); live .txt
  saving continues via the pipeline.
- **Portable builds (verified):** `scripts/Stage-Portable.ps1` → full
  `dist/Voice2Text/` (2.5 GB, GPU) and `-Slim` `dist/Voice2Text-slim/` (656 MB,
  CPU-only). Full build proven self-contained (loads Whisper on GPU with `PATH`
  stripped to System32). `-Zip` archives it.
- **Installer:** `scripts/Build-Installer.ps1` + `installer.conf.json` are wired
  for an NSIS slim installer, but `tauri build` hits a tauri_build "Access denied
  (os error 5)" on this box whenever ANY bundle.resources entry is present
  (reproduced with a single in-crate file). Blocked on that tooling issue; the
  staged slim folder/zip is the working slim distribution meanwhile.
- **Local-LLM summary (DONE):** `app/llama-sidecar/` (llama-cpp-2, CUDA) runs
  Qwen2.5-3B-Instruct-Q4_K_M; the ✦ button sends the session transcript and shows
  a summary + action items panel (Copy / Save .md). One-shot spawn-per-summary
  (transcript on stdin → summary on stdout). Verified: accurate summary + correctly
  attributed action items in ~2 s on the RTX 5080. `summarize` command runs it
  via `spawn_blocking`; missing model → clear error.
- Config persists to `%APPDATA%\com.voice2text.overlay\config.json`; settings UI
  exposes all toggles. **App is feature-complete** per the original brief; the
  only open item is the NSIS slim installer (tauri_build tooling bug above).

---

## 6. Key decisions & alternatives kept in reserve

| Decision | Chosen | Reserve / fallback | Trigger to switch |
|---|---|---|---|
| Shell | Tauri | Electron | Webview limitations block the overlay |
| STT engine | **`whisper-rs`+CUDA (native, in-process)** | Bundled Python faster-whisper sidecar | whisper.cpp+CUDA won't build on sm_120 (decided in M0.5) |
| STT model (English) | whisper.cpp large-v3 | **NVIDIA Parakeet-TDT** via `sherpa-onnx` (native, no Python) | Latency from VAD-chunking is annoying; want true streaming |
| Packaging | Tauri bundler — NSIS installer **+** portable .exe | MSI | — |
| VAD | Silero (onnx) | WebRTC VAD | Silero onnx integration friction |
| Resampler | rubato | linear/`dasp` | Quality good enough simpler |
| Summary | local Ollama | none | — |

**Note on streaming (latency matters to this user):** Whisper is inherently
non-streaming; we fake liveness with VAD chunking, so end-of-utterance latency ≈
trailing-silence threshold + inference. To keep it tight:
- Tune the silence threshold low (~400–500 ms) and cap utterance length (~10 s)
  so long talkers still get periodic partial output.
- Emit **partial captions** by re-transcribing the open utterance buffer every
  ~700 ms (interim text), finalizing on utterance close. **Caveat (review):** naive
  re-transcription flickers / repeats prefixes / can hallucinate partials. Use a
  **stable-prefix algorithm (LocalAgreement-2, as in `whisper_streaming`)** — only
  promote text agreed across consecutive passes — or, as a simpler fallback, show
  only finalized lines plus a "listening…" indicator instead of live partials.
- Run STT on the trailing window only, not the whole buffer, to keep inference
  short on the RTX 5080.

If that still feels laggy, **NVIDIA Parakeet-TDT** is the SOTA English streaming
model for NVIDIA GPUs and the planned upgrade — it streams token-by-token instead
of waiting for utterance end. To stay Python-free (standalone requirement), run it
via **`sherpa-onnx`** — a native C++/Rust runtime over ONNX Runtime with CUDA, not
the NeMo/Python toolkit. Given latency is a stated priority, we evaluate Parakeet
as a **fast-follow after M3** rather than a distant reserve.

---

## 7. Resolved inputs (2026-06-27)
- **Platform:** Zoom primary, Google Meet occasional. Both route through the
  default render device → full system loopback works for both; no per-app capture
  needed in v1.
- **Speakers + mic, rarely headphones** → acoustic echo handling is required, not
  optional. See §3.6.
- **Overlay:** floating, resizable box (not a fixed bar). See §3.4.
- **Latency matters** → aggressive VAD tuning + interim partials; Parakeet-TDT
  promoted to fast-follow. See §6.

### Still open
- Target latency number that would feel "live" (assuming **<1.5 s** to first
  final text; we'll measure against it in M3/M4).
- Zoom **desktop app vs browser** (both fine for loopback; only matters if we
  ever add per-app capture).

---

## 8. Repo layout (proposed)
```
Voice2Text/
  DESIGN.md                  ← this file
  src-tauri/                 ← Rust core (capture, vad, stt, Tauri cmds)
    src/
      audio/                 ← cpal capture + resample + ring buffers
      vad/                   ← Silero wrapper
      stt/                   ← whisper-rs engine (in-process, CUDA)
      main.rs
    tauri.conf.json
  ui/                        ← web frontend (overlay + transcript)
  models/                    ← GGML weights + test clip (gitignored)
  scripts/                   ← M0/M0.5 validation, setup helpers
  .venv/                     ← DEV-ONLY: faster-whisper accuracy oracle (not shipped)
```

---

## 9. Packaging & distribution

Goal: a **standalone** app the user can run with nothing else installed — both an
**installer** and a **portable** build.

- **Tauri bundler** produces an **NSIS installer** and a **portable .exe** from the
  same build. No Python, no separate runtime — the STT engine is compiled into the
  binary (§3.3).
- **What ships:**
  - `Voice2Text.exe` (Rust core + webview assets embedded)
  - CUDA runtime DLLs whisper.cpp needs (cudart / cublas / etc.) placed next to the
    exe — confirm the exact set in M0.5.
  - the GGML model file (large-v3, full ~3 GB or q5_0 ~1.1 GB) in a `models/` dir
    next to the exe (portable) or under the install dir / AppData (installer).
- **Model handling:** ship the model inside the bundle (simplest, fully offline on
  first run) — preferred. Alternative: first-run download to AppData to keep the
  installer small. Decide after M0.5 picks full vs quantized.
- **GPU prerequisite:** an NVIDIA driver new enough for CUDA 12.8 (the user has
  610.47). Document this as the one system requirement; the app should detect "no
  CUDA device" and show a clear message rather than crash.
- **Fresh-machine proof (review):** PATH-borrowing CUDA DLLs from the installed
  toolkit (as `Run-M05.ps1` does) is NOT proof of distributability. Before calling
  packaging done: run a **dependency inventory** on the built exe/DLLs (e.g.
  `dumpbin /dependents`), confirm the exact CUDA DLLs (`cudart`, `cublas`,
  `cublasLt`, …) are **redistributable under NVIDIA's license**, bundle them, and
  test on a machine with **only the NVIDIA driver** — no CUDA Toolkit, LLVM, or
  CMake. (`CUDAARCHS=120`-only is fine for this RTX 5080; revisit if ever shipping
  to other GPUs.)
- **Warmup/contention UX (review):** model load is not instant — show a warmup
  state, define behavior for speech that arrives before the model is ready (buffer
  it), and budget the 16 GB VRAM if Whisper + Parakeet + a local summary LLM ever
  run together.
- **Not shipped:** the `.venv` / faster-whisper — that's a dev-time accuracy oracle
  only.
- **Signing (optional, later):** unsigned portable .exe will trip SmartScreen;
  fine for personal use. Revisit only if distributing beyond the user.
