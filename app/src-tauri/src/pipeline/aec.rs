//! Acoustic echo cancellation (signal layer).
//!
//! The attendees' audio plays out the speakers and is re-captured by the mic, so
//! the mic stream carries an echo of the system (loopback) stream. This module
//! subtracts that echo from the mic *audio* before it ever reaches the streaming
//! recognizer — the complement of the text-level dedup in [`super::echo`], which
//! catches whatever survives.
//!
//! The two streams come from independent WASAPI clients on separate clocks, so
//! there is an unknown, slowly-drifting offset between them. We handle this in
//! two stages:
//!   1. a **cross-correlation delay estimator** finds the bulk alignment between
//!      the recent mic window and the buffered reference, re-checked periodically;
//!   2. a **normalized LMS (NLMS) adaptive filter** then models the short room
//!      impulse response on the aligned reference and subtracts the prediction.
//!
//! A simple double-talk guard freezes adaptation when the user is speaking over
//! the attendees, so the filter doesn't try to cancel (and distort) near-end
//! speech. This is a best-effort canceller meant to be paired with the text
//! dedup, not a substitute for hardware/OS AEC.

/// Adaptive filter length (taps). 1024 @ 16 kHz ≈ 64 ms of room response, which
/// the bulk-delay alignment is responsible for centering the echo within.
const FILTER_LEN: usize = 1024;
/// Reference ring capacity (power of two). Must exceed the max bulk delay plus
/// the filter length and a correlation window. 32768 ≈ 2 s @ 16 kHz.
const RING: usize = 32768;
const RING_MASK: u64 = (RING as u64) - 1;
/// Widest bulk delay (reference leading mic) the estimator searches: 1 s.
const MAX_DELAY: i64 = 16_000;
/// Correlation window used for delay estimation (≈256 ms).
const CORR_WIN: usize = 4096;
/// Re-estimate the bulk delay this often (in mic samples processed); ≈0.5 s.
const REESTIMATE_EVERY: u64 = 8_000;
/// NLMS step size (0..2). Lower = slower but stabler convergence.
const MU: f32 = 0.5;
/// NLMS regularization, guards the divide when the reference is near-silent.
const EPS: f32 = 1e-3;
/// Reference must carry at least this much per-tap power for adaptation to run
/// (no echo to learn from silence).
const REF_ACTIVE: f32 = 1e-5;
/// Smoothing for the slow power estimates (echo level, convergence latch).
const PWR_SMOOTH: f32 = 0.01;
/// Smoothing for the *fast* mic-power estimate that triggers the double-talk
/// freeze. Must react within a few samples so the filter cannot diverge on
/// near-end during the detection lag (which is what drags the slow estimate up
/// and masks the double-talk).
const FAST_SMOOTH: f32 = 0.2;
/// Double-talk is declared when the residual error is at least this fraction of
/// the mic power: a converged filter leaves a tiny residual on echo-only, but
/// near-end speech passes straight into the residual. Detecting on the
/// residual/mic *ratio* (rather than mic vs estimated-echo) avoids the feedback
/// loop where throttling the filter drags its own detector threshold down.
const RESIDUAL_DT_FRAC: f32 = 0.5;
/// The filter is considered "converged" once it has removed at least this
/// fraction of the mic power (i.e. residual ≤ (1−frac)·mic). The double-talk
/// detector is only trusted after convergence — before that the residual is
/// ~the whole mic and every frame would look like double-talk, so we adapt
/// freely to bootstrap.
const CONVERGED_FRAC: f32 = 0.8;
/// Minimum reference-active samples adapted before the convergence latch may
/// fire. Guards against latching during the initial power ramp (when the slow
/// mic-power estimate still lags low and the ratio is met spuriously), which
/// would freeze a half-converged filter. ≈256 ms at 16 kHz.
const MIN_ADAPT: u64 = 4 * FILTER_LEN as u64;
/// After double-talk is detected, keep the filter frozen for this many samples
/// even if the instantaneous detector drops. Bridges the brief residual nulls
/// within near-end speech (and tone zero-crossings) that would otherwise
/// re-enable adaptation and let the filter chew into the near-end. ≈200 ms.
const HANGOVER: u32 = 3_200;
/// Minimum correlation quality to accept a new delay estimate over the old one.
const MIN_CORR: f32 = 0.3;

pub struct Aec {
    /// Reference (loopback) ring, absolute-indexed via `ref_count`.
    ref_ring: Vec<f32>,
    ref_count: u64,
    /// Recent mic ring, used only for delay estimation.
    mic_ring: Vec<f32>,
    mic_count: u64,
    /// Adaptive filter weights (echo path estimate).
    weights: Vec<f32>,
    /// Alignment offset: reference sample for mic index `n` is at ref index
    /// `n + align` (see module docs). `None` until first estimated.
    align: Option<i64>,
    /// Mic samples processed since the last delay re-estimation.
    since_estimate: u64,
    /// Smoothed mic and residual-error power. The slow pair drives the
    /// convergence latch; the fast pair triggers the double-talk freeze quickly
    /// enough that the filter can't diverge during the detection lag.
    mic_pwr: f32,
    mic_fast: f32,
    err_pwr: f32,
    err_fast: f32,
    /// Latches once the filter is tracking the echo; gates the double-talk
    /// detector so it can't deadlock the unconverged (zero-output) filter.
    converged: bool,
    /// Remaining hold-down samples after a double-talk detection (hangover).
    dt_hold: u32,
    /// Reference-active samples adapted since the last (re)alignment; the
    /// convergence latch waits for this so it can't fire during the power ramp.
    adapt_count: u64,
}

impl Aec {
    pub fn new() -> Self {
        Self {
            ref_ring: vec![0.0; RING],
            ref_count: 0,
            mic_ring: vec![0.0; RING],
            mic_count: 0,
            weights: vec![0.0; FILTER_LEN],
            align: None,
            since_estimate: 0,
            mic_pwr: 0.0,
            mic_fast: 0.0,
            err_pwr: 0.0,
            err_fast: 0.0,
            converged: false,
            dt_hold: 0,
            adapt_count: 0,
        }
    }

    /// Forget all buffered audio, the learned echo path, and the alignment.
    /// Called when capture pauses/stops so a new session re-aligns from scratch
    /// (the device clocks and bulk delay may have changed).
    pub fn reset(&mut self) {
        self.ref_ring.iter_mut().for_each(|s| *s = 0.0);
        self.mic_ring.iter_mut().for_each(|s| *s = 0.0);
        self.weights.iter_mut().for_each(|w| *w = 0.0);
        self.ref_count = 0;
        self.mic_count = 0;
        self.align = None;
        self.since_estimate = 0;
        self.mic_pwr = 0.0;
        self.mic_fast = 0.0;
        self.err_pwr = 0.0;
        self.err_fast = 0.0;
        self.converged = false;
        self.dt_hold = 0;
        self.adapt_count = 0;
    }

    /// Feed loopback (system) audio. Call as system chunks arrive, before the
    /// mic chunks they will echo into.
    pub fn push_reference(&mut self, samples: &[f32]) {
        for &s in samples {
            self.ref_ring[(self.ref_count & RING_MASK) as usize] = s;
            self.ref_count += 1;
        }
    }

    /// Cancel echo from a mic chunk, returning the cleaned samples.
    pub fn process_capture(&mut self, mic: &[f32]) -> Vec<f32> {
        // Stage the raw mic into its ring for delay estimation.
        for &s in mic {
            self.mic_ring[(self.mic_count & RING_MASK) as usize] = s;
            self.mic_count += 1;
        }

        // (Re)estimate bulk alignment when due or when we have none yet.
        self.since_estimate += mic.len() as u64;
        if self.align.is_none() || self.since_estimate >= REESTIMATE_EVERY {
            self.estimate_delay();
            self.since_estimate = 0;
        }
        let align = match self.align {
            Some(a) => a,
            None => return mic.to_vec(), // not enough data to align yet
        };

        // The mic samples just appended occupy absolute indices
        // [mic_count - mic.len(), mic_count).
        let start = self.mic_count - mic.len() as u64;
        let mut out = Vec::with_capacity(mic.len());
        for (i, &d) in mic.iter().enumerate() {
            let n = start + i as u64;
            out.push(self.cancel_sample(n, d, align));
        }
        out
    }

    /// Cancel one mic sample at absolute mic index `n`.
    fn cancel_sample(&mut self, n: u64, d: f32, align: i64) -> f32 {
        // Newest aligned reference index for this mic sample.
        let head = n as i64 + align;
        // Need the whole tap window present in the ring; otherwise pass through.
        if head < 0 || head - (FILTER_LEN as i64 - 1) < self.ref_oldest() as i64 || (head as u64) >= self.ref_count {
            return d;
        }

        // y_hat = Σ weights[k] * ref[head - k]; also accumulate reference power.
        let mut y_hat = 0.0f32;
        let mut x_pwr = 0.0f32;
        for k in 0..FILTER_LEN {
            let x = self.ref_ring[((head as u64 - k as u64) & RING_MASK) as usize];
            y_hat += self.weights[k] * x;
            x_pwr += x * x;
        }
        let e = d - y_hat;

        // Track mic and residual power (slow pair → convergence latch, fast pair
        // → double-talk trigger).
        self.mic_pwr += PWR_SMOOTH * (d * d - self.mic_pwr);
        self.mic_fast += FAST_SMOOTH * (d * d - self.mic_fast);
        self.err_pwr += PWR_SMOOTH * (e * e - self.err_pwr);
        self.err_fast += FAST_SMOOTH * (e * e - self.err_fast);

        let reference_active = x_pwr / FILTER_LEN as f32 > REF_ACTIVE;
        // Double-talk: once converged, a large residual relative to the mic means
        // near-end speech is present (the echo would otherwise be cancelled to a
        // small residual). Suppressed before convergence, where the residual is
        // legitimately the whole mic and would freeze the filter forever. A
        // hangover holds the freeze across brief residual dips within speech.
        if self.converged && self.err_fast > RESIDUAL_DT_FRAC * self.mic_fast {
            self.dt_hold = HANGOVER;
        }
        let double_talk = self.dt_hold > 0;
        self.dt_hold = self.dt_hold.saturating_sub(1);

        if reference_active && !double_talk {
            self.adapt_count += 1;
            // Converged once the filter has removed enough of the mic power.
            if !self.converged
                && self.adapt_count >= MIN_ADAPT
                && self.mic_pwr > REF_ACTIVE
                && self.err_pwr <= (1.0 - CONVERGED_FRAC) * self.mic_pwr
            {
                self.converged = true;
            }
            let norm = MU * e / (x_pwr + EPS);
            for k in 0..FILTER_LEN {
                let x = self.ref_ring[((head as u64 - k as u64) & RING_MASK) as usize];
                self.weights[k] += norm * x;
            }
        }
        e
    }

    /// Oldest absolute reference index still held in the ring.
    fn ref_oldest(&self) -> u64 {
        self.ref_count.saturating_sub(RING as u64)
    }

    /// Cross-correlate the most recent mic window against the reference at a
    /// range of bulk delays, updating `align` if a confident peak is found.
    fn estimate_delay(&mut self) {
        if self.mic_count < CORR_WIN as u64 || self.ref_count < CORR_WIN as u64 {
            return;
        }
        let win = CORR_WIN as u64;
        let mic_lo = self.mic_count - win; // window [mic_lo, mic_count)

        // Search align so that ref index (n + align) stays within the ring for
        // the whole window: centered on the count difference, looking back up to
        // MAX_DELAY (reference leading the mic).
        let base = self.ref_count as i64 - self.mic_count as i64;
        let hi = base;
        let lo = base - MAX_DELAY;

        // Precompute mic window energy.
        let mut mic_energy = 0.0f32;
        for j in 0..CORR_WIN {
            let m = self.mic_ring[(((mic_lo + j as u64) & RING_MASK)) as usize];
            mic_energy += m * m;
        }
        if mic_energy <= 0.0 {
            return;
        }

        let oldest = self.ref_oldest() as i64;
        let mut best_score = MIN_CORR;
        let mut best_align: Option<i64> = None;
        let mut a = lo;
        while a <= hi {
            // Require the whole correlated reference span to be in the ring.
            let r_lo = mic_lo as i64 + a;
            let r_hi = (self.mic_count - 1) as i64 + a;
            if r_lo < oldest || r_hi >= self.ref_count as i64 {
                a += 1;
                continue;
            }
            let mut dot = 0.0f32;
            let mut ref_energy = 0.0f32;
            for j in 0..win {
                let m = self.mic_ring[(((mic_lo + j) & RING_MASK)) as usize];
                let r = self.ref_ring[((((mic_lo + j) as i64 + a) as u64 & RING_MASK)) as usize];
                dot += m * r;
                ref_energy += r * r;
            }
            if ref_energy > 0.0 {
                let score = dot / (mic_energy.sqrt() * ref_energy.sqrt());
                if score > best_score {
                    best_score = score;
                    best_align = Some(a);
                }
            }
            a += 1;
        }

        if let Some(a) = best_align {
            // On a re-estimate that moves the alignment, the existing weights no
            // longer line up with the reference, so reset them to re-converge.
            if self.align != Some(a) {
                if self.align.is_some() {
                    for w in &mut self.weights {
                        *w = 0.0;
                    }
                    self.converged = false;
                    self.adapt_count = 0;
                }
                self.align = Some(a);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny deterministic LCG so tests don't depend on `rand` (and stay stable).
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0 // ~[-1, 1)
        }
    }

    /// A 300 Hz near-end test tone at absolute sample index `t`. Amplitude is
    /// set so the talk-over is unambiguous (clearly above the echo level).
    fn near_end_tone(t: usize) -> f32 {
        0.8 * (2.0 * std::f32::consts::PI * 300.0 * t as f32 / super::super::RATE as f32).sin()
    }

    fn power(xs: &[f32]) -> f32 {
        if xs.is_empty() {
            return 0.0;
        }
        xs.iter().map(|x| x * x).sum::<f32>() / xs.len() as f32
    }

    /// With a pure delayed-and-attenuated echo and no near-end, the canceller
    /// should drive the residual well below the input echo.
    #[test]
    fn cancels_pure_echo() {
        const DELAY: usize = 2000;
        const GAIN: f32 = 0.6;
        const FRAME: usize = 160;
        const FRAMES: usize = 600; // ~6 s

        let mut rng = Lcg(0x1234_5678);
        let mut aec = Aec::new();
        let mut reference = vec![0.0f32; FRAME * FRAMES + DELAY];
        for r in reference.iter_mut() {
            *r = rng.next_f32();
        }

        let mut residual_tail = Vec::new();
        let mut echo_tail = Vec::new();
        for f in 0..FRAMES {
            let base = f * FRAME;
            let ref_frame = &reference[base..base + FRAME];
            // mic[t] = GAIN * reference[t - DELAY]
            let mut mic_frame = vec![0.0f32; FRAME];
            for (i, m) in mic_frame.iter_mut().enumerate() {
                let t = base + i;
                if t >= DELAY {
                    *m = GAIN * reference[t - DELAY];
                }
            }
            aec.push_reference(ref_frame);
            let out = aec.process_capture(&mic_frame);
            if f >= FRAMES - 60 {
                // last ~0.6 s, after convergence
                residual_tail.extend_from_slice(&out);
                echo_tail.extend_from_slice(&mic_frame);
            }
        }

        let reduction_db = 10.0 * (power(&echo_tail) / power(&residual_tail)).log10();
        assert!(reduction_db > 12.0, "echo reduction only {reduction_db:.1} dB");
    }

    /// Near-end speech (added on top of the echo) must survive cancellation: the
    /// output should still track the near-end signal.
    #[test]
    fn preserves_near_end_speech() {
        const DELAY: usize = 1500;
        const GAIN: f32 = 0.6;
        const FRAME: usize = 160;
        const PREROLL: usize = 400; // echo-only frames to converge the filter
        const TEST: usize = 150;

        let mut rng = Lcg(0x0bad_f00d);
        let mut aec = Aec::new();
        let total = FRAME * (PREROLL + TEST) + DELAY;
        let mut reference = vec![0.0f32; total];
        for r in reference.iter_mut() {
            *r = rng.next_f32();
        }

        let mut near_end = Vec::new();
        let mut output = Vec::new();
        for f in 0..(PREROLL + TEST) {
            let base = f * FRAME;
            let ref_frame = &reference[base..base + FRAME];
            let mut mic_frame = vec![0.0f32; FRAME];
            for (i, m) in mic_frame.iter_mut().enumerate() {
                let t = base + i;
                let echo = if t >= DELAY { GAIN * reference[t - DELAY] } else { 0.0 };
                // Near-end tone present only during the test segment.
                let ne = if f >= PREROLL { near_end_tone(t) } else { 0.0 };
                *m = echo + ne;
            }
            aec.push_reference(ref_frame);
            let out = aec.process_capture(&mic_frame);
            if f >= PREROLL {
                for (i, &o) in out.iter().enumerate() {
                    near_end.push(near_end_tone(base + i));
                    output.push(o);
                }
            }
        }

        // Output should correlate strongly with the near-end signal we injected.
        let mut dot = 0.0f32;
        let mut eo = 0.0f32;
        let mut en = 0.0f32;
        for (o, n) in output.iter().zip(near_end.iter()) {
            dot += o * n;
            eo += o * o;
            en += n * n;
        }
        let corr = dot / (eo.sqrt() * en.sqrt());
        assert!(corr > 0.7, "near-end correlation only {corr:.2}");
    }
}
