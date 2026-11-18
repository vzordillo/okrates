//! Statistical hypothesis tests and exact finite-state CTMC solvers for verification suites.

/// Lanczos log-gamma approximation (`g = 7`, 9-term Chebyshev fit).
pub fn ln_gamma(z: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if z < 0.5 {
        return (std::f64::consts::PI / (std::f64::consts::PI * z).sin()).ln() - ln_gamma(1.0 - z);
    }
    let z = z - 1.0;
    let mut sum = C[0];
    for (i, c) in C.iter().enumerate().skip(1) {
        sum += c / (z + i as f64);
    }
    let t = z + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + sum.ln()
}

/// Regularized upper incomplete gamma `Q(a, x) = 1 - P(a, x)`.
pub fn gamma_q(a: f64, x: f64) -> f64 {
    assert!(a.is_finite() && a > 0.0, "gamma_q requires finite a > 0");
    assert!(!x.is_nan(), "gamma_q requires non-NaN x");
    if x <= 0.0 {
        return 1.0;
    }
    if !x.is_finite() {
        return 0.0;
    }
    if x < a + 1.0 {
        let mut term = 1.0;
        let mut sum = 1.0;
        for n in 1..=10_000 {
            term *= x / (a + n as f64);
            sum += term;
            if term <= sum * 1e-17 {
                break;
            }
        }
        let p = (a * x.ln() - x - ln_gamma(a + 1.0) + sum.ln()).exp();
        (1.0 - p).clamp(0.0, 1.0)
    } else {
        const TINY: f64 = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / TINY;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..10_000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < TINY {
                d = TINY;
            }
            c = b + an / c;
            if c.abs() < TINY {
                c = TINY;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-16 {
                break;
            }
        }
        ((a * x.ln() - x - ln_gamma(a)).exp() * h).clamp(0.0, 1.0)
    }
}

/// Upper `alpha` critical value of a chi-square distribution with `df` degrees of freedom.
pub fn chi2_upper_quantile(df: usize, alpha: f64) -> f64 {
    assert!(df >= 1, "chi2_upper_quantile requires df >= 1");
    assert!(
        alpha.is_finite() && alpha > 0.0 && alpha < 1.0,
        "chi2_upper_quantile requires alpha in (0, 1), got {alpha}"
    );
    let a = df as f64 / 2.0;
    let mut hi = (df as f64).max(1.0) * 10.0 + 50.0;
    while gamma_q(a, hi / 2.0) > alpha {
        hi *= 2.0;
    }
    let mut lo = 0.0;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if gamma_q(a, mid / 2.0) > alpha {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

pub fn chi2_statistic(observed: &[u64], expected_probs: &[f64]) -> f64 {
    assert!(
        !observed.is_empty(),
        "chi2_statistic requires non-empty bins"
    );
    assert_eq!(observed.len(), expected_probs.len());
    let total_u64: u64 = observed.iter().sum();
    assert!(
        total_u64 > 0,
        "chi2_statistic requires positive total count"
    );
    let prob_sum: f64 = expected_probs.iter().sum();
    assert!(
        expected_probs.iter().all(|&p| p.is_finite() && p >= 0.0) && (prob_sum - 1.0).abs() <= 1e-9,
        "expected_probs must be non-negative and sum to 1 (sum={prob_sum})"
    );
    let total = total_u64 as f64;
    observed
        .iter()
        .zip(expected_probs)
        .map(|(&obs, &p)| {
            let exp = total * p;
            if exp == 0.0 {
                if obs == 0 { 0.0 } else { f64::INFINITY }
            } else {
                (obs as f64 - exp).powi(2) / exp
            }
        })
        .sum()
}

/// One-sample Kolmogorov–Smirnov statistic `D_n = sup_x |F_n(x) - F(x)|`.
pub fn ks_statistic<F: Fn(f64) -> f64>(samples: &[f64], cdf: F) -> f64 {
    assert!(!samples.is_empty());
    assert!(
        samples.iter().all(|x| !x.is_nan()),
        "ks_statistic samples must not contain NaN"
    );
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len() as f64;
    let mut d_max = 0.0_f64;
    for (i, &x) in sorted.iter().enumerate() {
        let raw_f = cdf(x);
        assert!(!raw_f.is_nan(), "ks_statistic CDF returned NaN");
        let f_x = raw_f.clamp(0.0, 1.0);
        let d_lo = f_x - (i as f64) / n;
        let d_hi = ((i + 1) as f64) / n - f_x;
        d_max = d_max.max(d_lo.abs()).max(d_hi.abs());
    }
    d_max
}

/// Asymptotic Kolmogorov survival probability `P(sqrt(n) D_n >= lambda)` with Stephens' finite-`n` correction.
pub fn ks_pvalue(d: f64, n: usize) -> f64 {
    assert!(n >= 1, "ks_pvalue requires n >= 1");
    assert!(
        d.is_finite() && d >= 0.0,
        "ks_pvalue requires finite d >= 0, got {d}"
    );
    let root_n = (n as f64).sqrt();
    let lambda = (root_n + 0.12 + 0.11 / root_n) * d;
    if lambda <= 0.0 {
        return 1.0;
    }
    if lambda < 0.8 {
        // Jacobi theta dual series for fast small-lambda convergence.
        let coeff = (2.0 * std::f64::consts::PI).sqrt() / lambda;
        let base = -(std::f64::consts::PI * std::f64::consts::PI) / (8.0 * lambda * lambda);
        let mut sum = 0.0;
        for k in 1..=200 {
            let odd = (2 * k - 1) as f64;
            let term = (odd * odd * base).exp();
            sum += term;
            if term <= sum * 1e-16 {
                break;
            }
        }
        return (1.0 - coeff * sum).clamp(0.0, 1.0);
    }
    let mut sum = 0.0;
    for k in 1..=200 {
        let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
        let term = 2.0 * sign * (-2.0 * (k as f64).powi(2) * lambda * lambda).exp();
        sum += term;
        if term.abs() < 1e-16 {
            break;
        }
    }
    sum.clamp(0.0, 1.0)
}

fn is_strongly_connected(rates: &[Vec<f64>]) -> bool {
    let n = rates.len();
    if n <= 1 {
        return true;
    }
    let reachable = |reverse: bool| -> usize {
        let mut seen = vec![false; n];
        let mut stack = vec![0_usize];
        seen[0] = true;
        let mut count = 1_usize;
        while let Some(u) = stack.pop() {
            for (v, seen_v) in seen.iter_mut().enumerate() {
                if u != v && !*seen_v {
                    let edge = if reverse { rates[v][u] } else { rates[u][v] };
                    if edge > 0.0 {
                        *seen_v = true;
                        count += 1;
                        stack.push(v);
                    }
                }
            }
        }
        count
    };
    reachable(false) == n && reachable(true) == n
}

/// Solves `Q^T pi = 0`, `sum_i pi_i = 1` for an irreducible finite-state CTMC rate matrix `rates[i][j]` (`i != j`).
pub fn solve_stationary_ctmc(rates: &[Vec<f64>]) -> Vec<f64> {
    let n = rates.len();
    assert!(n >= 1, "solve_stationary_ctmc requires at least one state");
    let mut rate_scale = 0.0_f64;
    for (i, row) in rates.iter().enumerate() {
        assert_eq!(row.len(), n);
        assert!(
            row[i].is_finite(),
            "CTMC diagonal entry rates[{i}][{i}] must be finite, got {}",
            row[i]
        );
        for (j, &r) in row.iter().enumerate() {
            if i != j {
                assert!(
                    r.is_finite() && r >= 0.0,
                    "CTMC off-diagonal rate rates[{i}][{j}] must be finite and non-negative, got {r}"
                );
                rate_scale = rate_scale.max(r);
            }
        }
    }
    if n == 1 {
        return vec![1.0];
    }
    assert!(
        rate_scale > 0.0 && is_strongly_connected(rates),
        "CTMC rate matrix is reducible"
    );

    let mut qt = vec![vec![0.0; n]; n];
    let mut a = vec![vec![0.0; n + 1]; n + 1];
    for i in 0..n {
        let mut out_sum = 0.0;
        for (j, &r) in rates[i].iter().enumerate() {
            if i != j {
                let r_norm = r / rate_scale;
                out_sum += r_norm;
                qt[j][i] += r_norm;
                a[j][i] += r_norm;
            }
        }
        qt[i][i] -= out_sum;
        a[i][i] -= out_sum;
    }
    for cell in a[n].iter_mut().take(n) {
        *cell = 1.0;
    }
    a[n][n] = 1.0;

    for col in 0..n {
        let mut pivot = col;
        for row in (col + 1)..=n {
            if a[row][col].abs() > a[pivot][col].abs() {
                pivot = row;
            }
        }
        a.swap(col, pivot);
        let diag = a[col][col];
        assert!(diag.abs() > 1e-13, "singular CTMC generator");
        for val in &mut a[col][col..=n] {
            *val /= diag;
        }
        let pivot_row = a[col][col..=n].to_vec();
        for (row_idx, row) in a.iter_mut().enumerate() {
            if row_idx != col {
                let factor = row[col];
                for (dst, &src) in row[col..=n].iter_mut().zip(&pivot_row) {
                    *dst -= factor * src;
                }
            }
        }
    }

    let mut pi: Vec<f64> = (0..n)
        .map(|i| {
            let v = a[i][n];
            assert!(v >= -1e-12, "negative stationary weight pi[{i}] = {v}");
            v.max(0.0)
        })
        .collect();
    let norm_sum: f64 = pi.iter().sum();
    for p in &mut pi {
        *p /= norm_sum;
    }
    let residual = qt
        .iter()
        .map(|row| row.iter().zip(&pi).map(|(q, p)| q * p).sum::<f64>().abs())
        .fold(0.0_f64, f64::max);
    assert!(
        residual <= 1e-11,
        "CTMC stationary residual {residual:e} exceeds tolerance"
    );
    pi
}

#[derive(Clone, Copy, Debug, Default)]
struct Dd(f64, f64);

#[inline]
fn dd_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    if !s.is_finite() {
        return (s, 0.0);
    }
    let a_prime = s - b;
    let b_prime = s - a_prime;
    (s, (a - a_prime) + (b - b_prime))
}

#[inline]
fn dd_two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    if p == 0.0 || !p.is_finite() {
        (p, 0.0)
    } else {
        (p, a.mul_add(b, -p))
    }
}

impl Dd {
    const ZERO: Self = Self(0.0, 0.0);

    #[inline]
    fn from_f64(x: f64) -> Self {
        Self(x, 0.0)
    }

    #[inline]
    fn neg(self) -> Self {
        Self(-self.0, -self.1)
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let (s, e) = dd_two_sum(self.0, rhs.0);
        let (s2, e2) = dd_two_sum(self.1, rhs.1);
        let (hi, lo) = dd_two_sum(s, e + s2);
        let (hi, lo) = dd_two_sum(hi, lo + e2);
        Self(hi, lo)
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let (p, e) = dd_two_prod(self.0, rhs.0);
        let (hi, lo) = dd_two_sum(p, e + self.0 * rhs.1 + self.1 * rhs.0);
        Self(hi, lo)
    }

    #[inline]
    fn div_u32(self, k: u32) -> Self {
        let kf = k as f64;
        let q1 = self.0 / kf;
        let (p, e) = dd_two_prod(q1, kf);
        let rem = (self.0 - p) - e + self.1;
        let (hi, lo) = dd_two_sum(q1, rem / kf);
        Self(hi, lo)
    }

    #[inline]
    fn scale_pow2(self, f: f64) -> Self {
        Self(self.0 * f, self.1 * f)
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self.0 + self.1
    }
}

fn dd_mat_mul(a: &[Vec<Dd>], b: &[Vec<Dd>]) -> Vec<Vec<Dd>> {
    let n = a.len();
    let mut out = vec![vec![Dd::ZERO; n]; n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i][k];
            if aik.0 == 0.0 && aik.1 == 0.0 {
                continue;
            }
            for j in 0..n {
                out[i][j] = out[i][j].add(aik.mul(b[k][j]));
            }
        }
    }
    out
}

fn reanchor_zero_row_sums(m: &mut [Vec<Dd>]) {
    for (i, row) in m.iter_mut().enumerate() {
        let mut off = Dd::ZERO;
        for (j, &cell) in row.iter().enumerate() {
            if i != j {
                off = off.add(cell);
            }
        }
        row[i] = off.neg();
    }
}

fn expm1_scaled_dd(mut scaled: Vec<Vec<Dd>>, scale_exp: u32, zero_row_sums: bool) -> Vec<Vec<Dd>> {
    let n = scaled.len();
    if zero_row_sums {
        reanchor_zero_row_sums(&mut scaled);
    }
    let mut a = scaled.clone();
    let mut term = scaled.clone();
    for k in 2..=25 {
        let next = dd_mat_mul(&term, &scaled);
        for i in 0..n {
            for j in 0..n {
                term[i][j] = next[i][j].div_u32(k);
                a[i][j] = a[i][j].add(term[i][j]);
            }
        }
    }
    if zero_row_sums {
        reanchor_zero_row_sums(&mut a);
    }
    for _ in 0..scale_exp {
        let sq = dd_mat_mul(&a, &a);
        for i in 0..n {
            for j in 0..n {
                a[i][j] = a[i][j].scale_pow2(2.0).add(sq[i][j]);
            }
        }
        if zero_row_sums {
            reanchor_zero_row_sums(&mut a);
        }
    }
    a
}

/// Matrix exponential `exp(m)` via rescaled scaling-and-squaring in double-double precision.
pub fn mat_exp(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = m.len();
    assert!(
        m.iter()
            .all(|row| row.len() == n && row.iter().all(|v| v.is_finite())),
        "mat_exp requires a square matrix with finite entries"
    );
    if n == 0 {
        return Vec::new();
    }
    let max_entry = m.iter().flatten().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    if max_entry == 0.0 {
        let mut id = vec![vec![0.0; n]; n];
        for (i, row) in id.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        return id;
    }
    let rescaled_norm = m
        .iter()
        .map(|row| row.iter().map(|&v| v.abs() / max_entry).sum::<f64>())
        .fold(0.0_f64, f64::max);
    let log2_norm = max_entry.log2() + rescaled_norm.log2();
    let scale_exp = if log2_norm > -1.0 {
        (log2_norm + 1.0).ceil().max(0.0) as u32
    } else {
        0
    };
    let s1 = scale_exp.min(1022) as i32;
    let s2 = (scale_exp - s1 as u32).min(1022) as i32;
    let f1 = 0.5_f64.powi(s1);
    let f2 = 0.5_f64.powi(s2);
    let pre = if max_entry > 1e290 {
        f64::MIN_POSITIVE
    } else {
        1.0
    };
    let zero_row_sums = m.iter().enumerate().all(|(i, row)| {
        row[i] <= 0.0 && {
            let mut off = Dd::ZERO;
            let mut nonneg = true;
            for (j, &v) in row.iter().enumerate() {
                if i != j {
                    if v < 0.0 {
                        nonneg = false;
                        break;
                    }
                    off = off.add(Dd::from_f64(v * pre));
                }
            }
            nonneg && off.add(Dd::from_f64(row[i] * pre)).to_f64().abs() <= f64::EPSILON * off.0
        }
    });
    let scaled: Vec<Vec<Dd>> = m
        .iter()
        .map(|row| row.iter().map(|&v| Dd::from_f64((v * f1) * f2)).collect())
        .collect();
    let a = expm1_scaled_dd(scaled, scale_exp, zero_row_sums);
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    if i == j {
                        a[i][j].add(Dd::from_f64(1.0)).to_f64()
                    } else {
                        a[i][j].to_f64()
                    }
                })
                .collect()
        })
        .collect()
}

/// Exact phase-type absorption CDF `F(t) = 1 - alpha_0^T exp(t S) 1` on transient states.
pub fn absorbing_escape_cdf(
    transient_rates: &[Vec<f64>],
    exit_rates: &[f64],
    initial_dist: &[f64],
    t: f64,
) -> f64 {
    assert!(t.is_finite(), "absorbing_escape_cdf requires finite t");
    let n = transient_rates.len();
    assert!(
        n >= 1,
        "absorbing_escape_cdf requires at least one transient state"
    );
    assert_eq!(exit_rates.len(), n);
    assert_eq!(initial_dist.len(), n);
    let mut max_rate = 0.0_f64;
    for i in 0..n {
        assert_eq!(transient_rates[i].len(), n);
        assert!(
            exit_rates[i].is_finite() && exit_rates[i] >= 0.0,
            "exit_rates[{i}] must be finite and non-negative"
        );
        assert!(
            initial_dist[i].is_finite() && initial_dist[i] >= 0.0,
            "initial_dist[{i}] must be finite and non-negative"
        );
        max_rate = max_rate.max(exit_rates[i]);
        for (j, &r) in transient_rates[i].iter().enumerate() {
            if i != j {
                assert!(
                    r.is_finite() && r >= 0.0,
                    "transient_rates[{i}][{j}] must be finite and non-negative"
                );
                max_rate = max_rate.max(r);
            }
        }
    }
    let dist_sum: f64 = initial_dist.iter().sum();
    assert!(
        (dist_sum - 1.0).abs() <= 1e-9,
        "initial_dist must sum to 1, got {dist_sum}"
    );
    if t <= 0.0 || max_rate == 0.0 {
        return 0.0;
    }
    let rescaled_row_max = (0..n)
        .map(|i| {
            let off: f64 = transient_rates[i]
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &r)| r / max_rate)
                .sum();
            2.0 * (off + exit_rates[i] / max_rate)
        })
        .fold(0.0_f64, f64::max);
    let log2_norm = t.log2() + max_rate.log2() + rescaled_row_max.log2();
    let scale_exp = if log2_norm > -1.0 {
        (log2_norm + 1.0).ceil().max(0.0) as u32
    } else {
        0
    };
    let s1 = scale_exp.min(1022) as i32;
    let rem = scale_exp - s1 as u32;
    let s2 = rem.min(1022) as i32;
    let s3 = (rem - s2 as u32).min(1022) as i32;
    let f1 = 0.5_f64.powi(s1);
    let f2 = 0.5_f64.powi(s2);
    let f3 = 0.5_f64.powi(s3);
    let scale_rate = |r: f64| -> Dd {
        let prod = r * t;
        if prod.is_finite() {
            let (p, e) = dd_two_prod(r, t);
            Dd::from_f64(p)
                .add(Dd::from_f64(e))
                .scale_pow2(f1)
                .scale_pow2(f2)
                .scale_pow2(f3)
        } else {
            let (p, e) = dd_two_prod(r * f1, t * f2);
            Dd(p * f3, e * f3)
        }
    };
    let m = n + 1;
    let mut scaled = vec![vec![Dd::ZERO; m]; m];
    for i in 0..n {
        for (j, &r) in transient_rates[i].iter().enumerate() {
            if i != j && r > 0.0 {
                scaled[i][j] = scale_rate(r);
            }
        }
        if exit_rates[i] > 0.0 {
            scaled[i][n] = scale_rate(exit_rates[i]);
        }
    }
    let a = expm1_scaled_dd(scaled, scale_exp, true);
    let mut absorbed = Dd::ZERO;
    for (i, &p0) in initial_dist.iter().enumerate() {
        absorbed = absorbed.add(a[i][n].mul(Dd::from_f64(p0)));
    }
    absorbed.to_f64().clamp(0.0, 1.0)
}
