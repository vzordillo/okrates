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
    if x <= 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        let mut term = 1.0;
        let mut sum = 1.0;
        let mut n = 1.0;
        while term > sum * 1e-17 {
            term *= x / (a + n);
            sum += term;
            n += 1.0;
        }
        let p = (a * x.ln() - x - ln_gamma(a + 1.0) + sum.ln()).exp();
        1.0 - p
    } else {
        const TINY: f64 = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / TINY;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..1000 {
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
        (a * x.ln() - x - ln_gamma(a)).exp() * h
    }
}

/// Upper `alpha` critical value of a chi-square distribution with `df` degrees of freedom.
pub fn chi2_upper_quantile(df: usize, alpha: f64) -> f64 {
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
    assert_eq!(observed.len(), expected_probs.len());
    let total: f64 = observed.iter().sum::<u64>() as f64;
    observed
        .iter()
        .zip(expected_probs)
        .map(|(&obs, &p)| {
            let exp = total * p;
            (obs as f64 - exp).powi(2) / exp
        })
        .sum()
}

/// One-sample Kolmogorov–Smirnov statistic `D_n = sup_x |F_n(x) - F(x)|`.
pub fn ks_statistic<F: Fn(f64) -> f64>(samples: &[f64], cdf: F) -> f64 {
    assert!(!samples.is_empty());
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len() as f64;
    let mut d_max = 0.0_f64;
    for (i, &x) in sorted.iter().enumerate() {
        let f_x = cdf(x).clamp(0.0, 1.0);
        let d_lo = f_x - (i as f64) / n;
        let d_hi = ((i + 1) as f64) / n - f_x;
        d_max = d_max.max(d_lo.abs()).max(d_hi.abs());
    }
    d_max
}

/// Asymptotic Kolmogorov survival probability `P(sqrt(n) D_n >= lambda)` with Stephens' finite-`n` correction.
pub fn ks_pvalue(d: f64, n: usize) -> f64 {
    let root_n = (n as f64).sqrt();
    let lambda = (root_n + 0.12 + 0.11 / root_n) * d;
    if lambda <= 0.0 {
        return 1.0;
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

/// Solves `Q^T pi = 0`, `sum_i pi_i = 1` for an irreducible finite-state CTMC rate matrix `rates[i][j]` (`i != j`).
pub fn solve_stationary_ctmc(rates: &[Vec<f64>]) -> Vec<f64> {
    let n = rates.len();
    assert!(n >= 1);
    let mut a = vec![vec![0.0; n + 1]; n];
    for i in 0..n {
        assert_eq!(rates[i].len(), n);
        let mut out_sum = 0.0;
        for (j, &r) in rates[i].iter().enumerate() {
            if i != j {
                out_sum += r;
                a[j][i] += r;
            }
        }
        a[i][i] -= out_sum;
    }
    // Replace final row with normalization constraint sum_j pi_j = 1.
    for cell in a[n - 1].iter_mut().take(n) {
        *cell = 1.0;
    }
    a[n - 1][n] = 1.0;

    for col in 0..n {
        let mut pivot = col;
        for row in (col + 1)..n {
            if a[row][col].abs() > a[pivot][col].abs() {
                pivot = row;
            }
        }
        a.swap(col, pivot);
        let diag = a[col][col];
        assert!(diag.abs() > 1e-15, "singular CTMC generator");
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
    (0..n).map(|i| a[i][n]).collect()
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut out = vec![vec![0.0; n]; n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i][k];
            for j in 0..n {
                out[i][j] += aik * b[k][j];
            }
        }
    }
    out
}

/// Matrix exponential `exp(m)` via scaling-and-squaring with an 18-term Taylor series.
pub fn mat_exp(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = m.len();
    let norm = m
        .iter()
        .map(|row| row.iter().map(|v| v.abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);
    let scale_exp = if norm > 0.5 {
        (norm * 2.0).log2().ceil().max(0.0) as u32
    } else {
        0
    };
    let scale = 2.0_f64.powi(-(scale_exp as i32));
    let scaled: Vec<Vec<f64>> = m
        .iter()
        .map(|row| row.iter().map(|&v| v * scale).collect())
        .collect();

    let mut sum = vec![vec![0.0; n]; n];
    let mut term = vec![vec![0.0; n]; n];
    for i in 0..n {
        sum[i][i] = 1.0;
        term[i][i] = 1.0;
    }
    for k in 1..=20 {
        let next = mat_mul(&term, &scaled);
        let inv_k = 1.0 / (k as f64);
        for i in 0..n {
            for j in 0..n {
                term[i][j] = next[i][j] * inv_k;
                sum[i][j] += term[i][j];
            }
        }
    }
    for _ in 0..scale_exp {
        sum = mat_mul(&sum, &sum);
    }
    sum
}

/// Exact phase-type absorption CDF `F(t) = 1 - alpha_0^T exp(t S) 1` on transient states.
pub fn absorbing_escape_cdf(
    transient_rates: &[Vec<f64>],
    exit_rates: &[f64],
    initial_dist: &[f64],
    t: f64,
) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    let n = transient_rates.len();
    assert_eq!(exit_rates.len(), n);
    assert_eq!(initial_dist.len(), n);
    let mut ts = vec![vec![0.0; n]; n];
    for i in 0..n {
        let mut out_sum = exit_rates[i];
        for (j, &r) in transient_rates[i].iter().enumerate() {
            if i != j {
                out_sum += r;
                ts[i][j] = t * r;
            }
        }
        ts[i][i] = -t * out_sum;
    }
    let exp_ts = mat_exp(&ts);
    let mut survival = 0.0;
    for (i, &p0) in initial_dist.iter().enumerate() {
        let row_sum: f64 = exp_ts[i].iter().sum();
        survival += p0 * row_sum;
    }
    (1.0 - survival).clamp(0.0, 1.0)
}
