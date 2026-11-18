//! The running peak and time-weighted mean of a count, shared by the `fecr_ris_pilot`
//! example and `tests/fecr_ris_pilot_tally.rs` (included with `#[path]`, not part of the
//! crate's API).

/// Running peak and time-weighted mean of a count.
#[derive(Default)]
pub struct Tally {
    pub peak: u64,
    weighted: f64,
    /// Neumaier compensation for `weighted`.
    weighted_comp: f64,
    /// The sum of the sampled waiting times seen so far.
    elapsed: f64,
    /// Neumaier compensation for `elapsed`.
    elapsed_comp: f64,
}

/// Neumaier-compensated `*sum += x`, keeping the rounding error in `*comp`.
fn add(sum: &mut f64, comp: &mut f64, x: f64) {
    let t = *sum + x;
    if sum.abs() >= x.abs() {
        *comp += (*sum - t) + x;
    } else {
        *comp += (x - t) + *sum;
    }
    *sum = t;
}

impl Tally {
    /// `before` held for `dt`; `after` is the count once the event has been applied.
    pub fn update(&mut self, before: u64, after: u64, dt: f64) {
        add(
            &mut self.weighted,
            &mut self.weighted_comp,
            before as f64 * dt,
        );
        add(&mut self.elapsed, &mut self.elapsed_comp, dt);
        self.peak = self.peak.max(after);
    }

    /// The time-weighted mean, or 0 when no time has elapsed.
    pub fn mean(&self) -> f64 {
        let elapsed = self.elapsed + self.elapsed_comp;
        if elapsed > 0.0 {
            (self.weighted + self.weighted_comp) / elapsed
        } else {
            0.0
        }
    }
}
