//! The running peak and time-weighted mean of a count, shared by the `fecr_ris_pilot`
//! example and `tests/fecr_ris_pilot_tally.rs` (included with `#[path]`, not part of the
//! crate's API).

/// Running peak and time-weighted mean of a count.
#[derive(Default)]
pub struct Tally {
    pub peak: u64,
    weighted: f64,
    /// The sum of the sampled waiting times seen so far.
    elapsed: f64,
}

impl Tally {
    /// `before` held for `dt`; `after` is the count once the event has been applied.
    pub fn update(&mut self, before: u64, after: u64, dt: f64) {
        self.weighted += before as f64 * dt;
        self.elapsed += dt;
        self.peak = self.peak.max(after);
    }

    /// The time-weighted mean, or 0 when no time has elapsed.
    pub fn mean(&self) -> f64 {
        if self.elapsed > 0.0 {
            self.weighted / self.elapsed
        } else {
            0.0
        }
    }
}
