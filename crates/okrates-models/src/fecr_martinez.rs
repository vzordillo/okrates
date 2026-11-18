//! Fe–Cr pair-interaction energetics of Martínez, Senninger, Fu, and Soisson,
//! *Phys. Rev. B* **86**, 224109 (2012) ("M12"), with Ω(x) from Levesque et al.,
//! arXiv:1307.7561 (PWSCF-PAW Table I).
//!
//! Model conventions:
//! - 1NN and 2NN bonds (`g_2NN = 0.5 * g_1NN`); Einstein + linear `c_P` from M12 Table I,
//!   with `h_AA = +2H/11`, `s_AA = +2S/11`, and `s_FeCr` carrying `-Omega(x)/(11 Theta)`.
//! - Local Cr concentration `x = (n_Cr(s) + n_Cr(t)) / 30` over `{s} ∪ 1NN ∪ 2NN`,
//!   treating vacancies and saddle-end empty sites as non-Cr.
//! - Saddle-point interactions `g_tilde[jumper][spectator]` are fitted at construction to
//!   reproduce pure-metal and dilute-solute migration free energies from M12 Table I.
//! - Because M12 has no global lower bound on `raw = E_saddle - E_initial`, [`BarrierFloorPolicy`]
//!   controls transitions where `raw < max(0, dE)` (default [`BarrierFloorPolicy::SymmetricFloor`]
//!   raises the saddle to `max(G_initial, G_final)` so forward/reverse detailed balance stays exact).

use std::error::Error;

use okrates_core::Checkpointable;

use crate::bcc::BccLattice;
use crate::fecr::{
    Energetics, FeCrError, FeCrModel, Species, decode_fecr_state, encode_fecr_common_params,
    encode_fecr_state,
};
use crate::vacancy::BOLTZMANN_EV_PER_K;

/// M12 and Levesque Table I constants (energies in eV, entropies in `k_B`, temperatures in K).
pub mod params {
    pub const H0_FE: f64 = -4.28;
    pub const H0_CR: f64 = -4.10;
    pub const CP_THETA_FE: f64 = 310.0;
    pub const CP_LINEAR_FE: f64 = 6.64e-4;
    pub const CP_THETA_CR: f64 = 360.0;
    pub const CP_LINEAR_CR: f64 = 5.0e-4;
    pub const DHF_V_FE: f64 = 2.18;
    pub const DHF_V_CR: f64 = 2.33;
    pub const DSF_V_FE: f64 = 4.1;
    pub const DSF_V_CR: f64 = 4.1;
    pub const THETA_K: f64 = 1480.0;
    pub const DH0_FE: f64 = 0.69;
    pub const DH2_FE: f64 = 0.57;
    pub const DH0_CR: f64 = 0.78;
    pub const DH2_CR: f64 = 0.70;
    pub const DS_MIG: f64 = 2.1;
    pub const OMEGA_ALPHA: f64 = 0.070;
    pub const OMEGA_BETA: f64 = -2.288;
    pub const OMEGA_GAMMA: f64 = 4.439;
    pub const OMEGA_DELTA: f64 = -2.480;
    pub const NU_FE: f64 = 1e13;
    pub const NU_CR: f64 = 1e13;
}

use params::*;

const LOCAL_DENOMINATOR: f64 = 30.0;
const MIN_CELLS: usize = 3;

/// Handling of M12 jumps whose raw pair-model barrier falls below `max(0, dE)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BarrierFloorPolicy {
    /// Clamps barrier to `max(raw, max(0, dE))`, preserving exact detailed balance.
    #[default]
    SymmetricFloor,
    /// Returns the raw pair-model barrier without clamping.
    Unfloored,
    /// Signals an invalid transition (`NaN` barrier -> `SimError::InvalidRate`) without panicking.
    Strict,
}

impl BarrierFloorPolicy {
    pub(crate) fn to_byte(self) -> u8 {
        match self {
            Self::SymmetricFloor => 0,
            Self::Unfloored => 1,
            Self::Strict => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cell {
    Fe,
    Cr,
    Vacancy,
    Empty,
}

impl From<Species> for Cell {
    fn from(species: Species) -> Self {
        match species {
            Species::Fe => Self::Fe,
            Species::Cr => Self::Cr,
            Species::Vacancy => Self::Vacancy,
        }
    }
}

#[derive(Clone, Copy)]
struct View<'a> {
    occupancy: &'a [Species],
    overrides: [Option<(usize, Cell)>; 2],
}

impl View<'_> {
    fn get(&self, site: usize) -> Cell {
        if let Some((s, cell)) = self.overrides[0]
            && s == site
        {
            return cell;
        }
        if let Some((s, cell)) = self.overrides[1]
            && s == site
        {
            return cell;
        }
        self.occupancy[site].into()
    }

    fn cr_count(&self, lattice: &BccLattice, site: usize) -> u32 {
        let occupancy = self.occupancy;
        let neighbors = lattice.neighbors(site);
        let second = lattice.second_neighbors(site);
        let mut n = u32::from(matches!(occupancy[site], Species::Cr));
        let mut k = 0;
        while k < 8 {
            n += u32::from(matches!(occupancy[neighbors[k]], Species::Cr));
            k += 1;
        }
        let mut k = 0;
        while k < 6 {
            n += u32::from(matches!(occupancy[second[k]], Species::Cr));
            k += 1;
        }
        let mut o = 0;
        while o < 2 {
            if let Some((k, cell)) = self.overrides[o] {
                let was = matches!(occupancy[k], Species::Cr);
                let now = cell == Cell::Cr;
                if was != now && (k == site || neighbors.contains(&k) || second.contains(&k)) {
                    if now {
                        n += 1;
                    } else {
                        n -= 1;
                    }
                }
            }
            o += 1;
        }
        n
    }
}

fn is_mixed(a: Cell, b: Cell) -> bool {
    matches!((a, b), (Cell::Fe, Cell::Cr) | (Cell::Cr, Cell::Fe))
}

fn omega(x: f64) -> f64 {
    (x - OMEGA_ALPHA) * (OMEGA_BETA * x * x + OMEGA_GAMMA * x + OMEGA_DELTA)
}

fn cp_integrals(theta: f64, linear: f64, t: f64) -> (f64, f64) {
    if t == 0.0 {
        return (0.0, 0.0);
    }
    let k = BOLTZMANN_EV_PER_K;
    let u = theta / t;
    let (einstein_h, einstein_s) = if u > 700.0 {
        (0.0, 0.0)
    } else {
        let e = u.exp_m1();
        (
            3.0 * k * theta / e,
            3.0 * k * (u / e - (-(-u).exp()).ln_1p()),
        )
    };
    (
        einstein_h + linear * k * t * t / 2.0,
        einstein_s + linear * k * t,
    )
}

fn region(lattice: &BccLattice, sites: &[usize]) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::with_capacity(15 * sites.len());
    let mut push = |site: usize| {
        let mut pos = out.len();
        while pos > 0 && out[pos - 1] >= site {
            if out[pos - 1] == site {
                return;
            }
            pos -= 1;
        }
        out.insert(pos, site);
    };
    for &s in sites {
        push(s);
        let (nb, sn) = (lattice.neighbors(s), lattice.second_neighbors(s));
        let mut k = 0;
        while k < 8 {
            push(nb[k]);
            k += 1;
        }
        let mut k = 0;
        while k < 6 {
            push(sn[k]);
            k += 1;
        }
    }
    out
}

#[derive(Clone, Debug)]
pub struct Martinez2012 {
    temperature_k: f64,
    nu: [f64; 2],
    saddle: [[f64; 2]; 2],
    g_aa: [f64; 2],
    g_av: [f64; 2],
    omega_scale: f64,
    policy: BarrierFloorPolicy,
    floor_activations: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl Martinez2012 {
    /// Creates M12 energetics at `temperature_k >= 0` (`0 K` allowed for static checks) using [`BarrierFloorPolicy::SymmetricFloor`].
    pub fn new(temperature_k: f64) -> Result<Self, FeCrError> {
        Self::with_policy(
            temperature_k,
            NU_FE,
            NU_CR,
            BarrierFloorPolicy::SymmetricFloor,
        )
    }

    pub fn with_attempt_frequencies(
        temperature_k: f64,
        nu_fe: f64,
        nu_cr: f64,
    ) -> Result<Self, FeCrError> {
        Self::with_policy(
            temperature_k,
            nu_fe,
            nu_cr,
            BarrierFloorPolicy::SymmetricFloor,
        )
    }

    pub fn with_policy(
        temperature_k: f64,
        nu_fe: f64,
        nu_cr: f64,
        policy: BarrierFloorPolicy,
    ) -> Result<Self, FeCrError> {
        for nu in [nu_fe, nu_cr] {
            if !nu.is_finite() || nu <= 0.0 {
                return Err(FeCrError::AttemptFrequency(nu));
            }
        }
        if !temperature_k.is_finite() || temperature_k < 0.0 {
            return Err(FeCrError::Temperature(temperature_k));
        }
        let t = temperature_k;
        let k = BOLTZMANN_EV_PER_K;
        let gibbs = |h0: f64, theta: f64, linear: f64| {
            let (ih, is) = cp_integrals(theta, linear, t);
            h0 + ih - t * is
        };
        let g_aa = [
            2.0 * gibbs(H0_FE, CP_THETA_FE, CP_LINEAR_FE) / 11.0,
            2.0 * gibbs(H0_CR, CP_THETA_CR, CP_LINEAR_CR) / 11.0,
        ];
        let g_av = [
            (DHF_V_FE - t * k * DSF_V_FE) / 11.0 + g_aa[0] / 2.0,
            (DHF_V_CR - t * k * DSF_V_CR) / 11.0 + g_aa[1] / 2.0,
        ];
        let mut model = Self {
            temperature_k,
            nu: [nu_fe, nu_cr],
            saddle: [[0.0; 2]; 2],
            g_aa,
            g_av,
            omega_scale: (1.0 - t / THETA_K) / 11.0,
            policy,
            floor_activations: std::sync::Arc::default(),
        };
        model.fit_saddle();
        Ok(model)
    }

    fn fit_saddle(&mut self) {
        let lattice = BccLattice::new([6, 6, 6]).expect("6x6x6 is a valid lattice");
        let site = |p: [i64; 3]| lattice.site_at(p).expect("valid BCC position");
        let ts = self.temperature_k * BOLTZMANN_EV_PER_K * DS_MIG;
        let dh0 = [DH0_FE, DH0_CR];
        let dh2 = [DH2_FE, DH2_CR];
        let vacancy = site([5, 5, 5]);
        let species = [Species::Fe, Species::Cr];
        let mut saddle = [[0.0; 2]; 2];
        for m in 0..2 {
            let solute = 1 - m;
            let mut occupancy = vec![species[m]; lattice.site_count()];
            occupancy[vacancy] = Species::Vacancy;
            let jumper = site([6, 4, 4]);
            let b = self.barrier_raw(&lattice, &occupancy, vacancy, jumper);
            saddle[m][m] = (dh0[m] - ts - b) / 6.0;

            let jumper = site([4, 4, 4]);
            occupancy[jumper] = species[solute];
            let b = self.barrier_raw(&lattice, &occupancy, vacancy, jumper);
            saddle[solute][m] = (dh2[m] - ts - b) / 6.0;
        }
        self.saddle = saddle;
    }

    pub fn policy(&self) -> BarrierFloorPolicy {
        self.policy
    }

    pub fn attempt_frequencies(&self) -> [f64; 2] {
        self.nu
    }

    /// Saddle interactions `g_tilde[jumper][neighbour]` in eV (`Fe = 0`, `Cr = 1`).
    pub fn saddle_interactions(&self) -> [[f64; 2]; 2] {
        self.saddle
    }

    /// Count of [`Energetics::barrier`] evaluations where `raw < max(0, dE)` (shared across clones).
    pub fn floor_activations(&self) -> u64 {
        self.floor_activations
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn reset_floor_activations(&self) {
        self.floor_activations
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Unfloored `E_saddle - E_initial` in eV (`NaN` if sites violate single-vacancy invariants).
    pub fn raw_barrier(
        &self,
        lattice: &BccLattice,
        occupancy: &[Species],
        vacancy_site: usize,
        atom_site: usize,
    ) -> f64 {
        self.barrier_raw(lattice, occupancy, vacancy_site, atom_site)
    }

    #[doc(hidden)]
    pub fn with_saddle_shift(mut self, shift_ev: f64) -> Self {
        for row in &mut self.saddle {
            for g in row {
                *g += shift_ev;
            }
        }
        self
    }

    pub fn temperature_k(&self) -> f64 {
        self.temperature_k
    }

    fn bond(&self, a: Cell, b: Cell, n_sum: u32) -> f64 {
        use Cell::{Cr, Empty, Fe, Vacancy};
        match (a, b) {
            (Empty, _) | (_, Empty) | (Vacancy, Vacancy) => 0.0,
            (Vacancy, Fe) | (Fe, Vacancy) => self.g_av[0],
            (Vacancy, Cr) | (Cr, Vacancy) => self.g_av[1],
            (Fe, Fe) => self.g_aa[0],
            (Cr, Cr) => self.g_aa[1],
            (Fe, Cr) | (Cr, Fe) => {
                let x = f64::from(n_sum) / LOCAL_DENOMINATOR;
                0.5 * (self.g_aa[0] + self.g_aa[1]) - omega(x) * self.omega_scale
            }
        }
    }

    /// Exact per-bond difference over `region` so unchanged bulk bonds cancel identically to zero.
    fn region_difference(
        &self,
        lattice: &BccLattice,
        before: &View,
        after: &View,
        region: &[usize],
        changed: [usize; 2],
    ) -> f64 {
        let occupancy = before.occupancy;
        let mut sum = 0.0;
        let mut i = 0;
        while i < region.len() {
            let s = region[i];
            i += 1;
            let s_changed = s == changed[0] || s == changed[1];
            let species = occupancy[s];
            let (s_before, s_after) = (before.get(s), after.get(s));
            let (mut n_before, mut n_after) = (None, None);
            let mut shell_index = 0;
            while shell_index < 2 {
                let (shell, weight) = if shell_index == 0 {
                    (&lattice.neighbors(s)[..], 1.0)
                } else {
                    (&lattice.second_neighbors(s)[..], 0.5)
                };
                shell_index += 1;
                let mut k = 0;
                while k < shell.len() {
                    let t = shell[k];
                    k += 1;
                    if !s_changed
                        && t != changed[0]
                        && t != changed[1]
                        && !matches!(
                            (species, occupancy[t]),
                            (Species::Fe, Species::Cr) | (Species::Cr, Species::Fe)
                        )
                    {
                        continue;
                    }
                    if t < s && region.binary_search(&t).is_ok() {
                        continue;
                    }
                    let (t_before, t_after) = (before.get(t), after.get(t));
                    let n_sum_before = if is_mixed(s_before, t_before) {
                        *n_before.get_or_insert_with(|| before.cr_count(lattice, s))
                            + before.cr_count(lattice, t)
                    } else {
                        0
                    };
                    let n_sum_after = if is_mixed(s_after, t_after) {
                        *n_after.get_or_insert_with(|| after.cr_count(lattice, s))
                            + after.cr_count(lattice, t)
                    } else {
                        0
                    };
                    sum += weight
                        * (self.bond(s_after, t_after, n_sum_after)
                            - self.bond(s_before, t_before, n_sum_before));
                }
            }
        }
        sum
    }

    fn local_delta(
        &self,
        lattice: &BccLattice,
        occupancy: &[Species],
        sites: [usize; 2],
        after: [Cell; 2],
    ) -> f64 {
        let region = region(lattice, &sites);
        let before = View {
            occupancy,
            overrides: [None, None],
        };
        let after = View {
            occupancy,
            overrides: [Some((sites[0], after[0])), Some((sites[1], after[1]))],
        };
        self.region_difference(lattice, &before, &after, &region, sites)
    }

    fn barrier_raw(
        &self,
        lattice: &BccLattice,
        occupancy: &[Species],
        vacancy_site: usize,
        atom_site: usize,
    ) -> f64 {
        let (q, p) = (vacancy_site, atom_site);
        if q >= occupancy.len() || p >= occupancy.len() || occupancy[q] != Species::Vacancy {
            return f64::NAN;
        }
        let jumper = match occupancy[p] {
            Species::Fe => 0,
            Species::Cr => 1,
            Species::Vacancy => return f64::NAN,
        };
        let mut spectators = 0.0;
        let candidates = lattice
            .neighbors(p)
            .iter()
            .filter(|k| lattice.second_neighbors(q).contains(k))
            .chain(
                lattice
                    .second_neighbors(p)
                    .iter()
                    .filter(|k| lattice.neighbors(q).contains(k)),
            );
        for &k in candidates {
            if k == p || k == q {
                continue;
            }
            let neighbour = match occupancy[k] {
                Species::Fe => 0,
                Species::Cr => 1,
                Species::Vacancy => return f64::NAN,
            };
            spectators += self.saddle[jumper][neighbour];
        }
        spectators + self.local_delta(lattice, occupancy, [p, q], [Cell::Empty, Cell::Empty])
    }
}

impl Energetics for Martinez2012 {
    fn total_energy(&self, lattice: &BccLattice, occupancy: &[Species]) -> f64 {
        let view = View {
            occupancy,
            overrides: [None, None],
        };
        let n: Vec<u32> = (0..occupancy.len())
            .map(|s| view.cr_count(lattice, s))
            .collect();
        let mut sum = 0.0;
        for s in 0..occupancy.len() {
            let a = Cell::from(occupancy[s]);
            for (shell, weight) in [
                (&lattice.neighbors(s)[..], 1.0),
                (&lattice.second_neighbors(s)[..], 0.5),
            ] {
                for &t in shell {
                    if t > s {
                        sum += weight * self.bond(a, occupancy[t].into(), n[s] + n[t]);
                    }
                }
            }
        }
        sum
    }

    fn exchange_delta(
        &self,
        lattice: &BccLattice,
        occupancy: &[Species],
        vacancy_site: usize,
        atom_site: usize,
    ) -> f64 {
        if vacancy_site >= occupancy.len() || atom_site >= occupancy.len() {
            return f64::NAN;
        }
        self.local_delta(
            lattice,
            occupancy,
            [vacancy_site, atom_site],
            [occupancy[atom_site].into(), Cell::Vacancy],
        )
    }

    fn swap_delta(&self, lattice: &BccLattice, occupancy: &[Species], i: usize, j: usize) -> f64 {
        if i >= occupancy.len() || j >= occupancy.len() {
            return f64::NAN;
        }
        self.local_delta(
            lattice,
            occupancy,
            [i, j],
            [occupancy[j].into(), occupancy[i].into()],
        )
    }

    fn barrier(
        &self,
        lattice: &BccLattice,
        occupancy: &[Species],
        vacancy_site: usize,
        atom_site: usize,
    ) -> f64 {
        let raw = self.barrier_raw(lattice, occupancy, vacancy_site, atom_site);
        let de = self.exchange_delta(lattice, occupancy, vacancy_site, atom_site);
        if !raw.is_finite() || !de.is_finite() {
            return f64::NAN;
        }
        let floor = de.max(0.0);
        if raw < floor {
            self.floor_activations
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return match self.policy {
                BarrierFloorPolicy::SymmetricFloor => floor,
                BarrierFloorPolicy::Unfloored => raw,
                BarrierFloorPolicy::Strict => f64::NAN,
            };
        }
        raw
    }

    fn attempt_frequency(&self, atom: Species) -> Option<f64> {
        match atom {
            Species::Fe => Some(self.nu[0]),
            Species::Cr => Some(self.nu[1]),
            Species::Vacancy => None,
        }
    }

    fn validate(&self, lattice: &BccLattice, temperature_k: f64) -> Result<(), FeCrError> {
        if lattice.cells().iter().any(|&c| c < MIN_CELLS) {
            return Err(FeCrError::LatticeTooSmall {
                min_cells: MIN_CELLS,
            });
        }
        if temperature_k != self.temperature_k {
            return Err(FeCrError::TemperatureMismatch {
                model: temperature_k,
                energetics: self.temperature_k,
            });
        }
        Ok(())
    }
}

impl Checkpointable for FeCrModel<Martinez2012> {
    const MODEL_ID: &'static str = "okrates.fecr_martinez2012";

    fn encode_params(&self, out: &mut Vec<u8>) {
        encode_fecr_common_params(self, out);
        let e = self.energetics();
        out.extend_from_slice(&e.temperature_k.to_bits().to_le_bytes());
        for &nu in &e.nu {
            out.extend_from_slice(&nu.to_bits().to_le_bytes());
        }
        for row in &e.saddle {
            for &g in row {
                out.extend_from_slice(&g.to_bits().to_le_bytes());
            }
        }
        out.push(e.policy.to_byte());
    }

    fn encode_state(&self, state: &Self::State, out: &mut Vec<u8>) {
        encode_fecr_state(state, out);
    }

    fn decode_state(&self, bytes: &[u8]) -> Result<Self::State, Box<dyn Error + Send + Sync>> {
        decode_fecr_state(self, bytes)
    }
}
