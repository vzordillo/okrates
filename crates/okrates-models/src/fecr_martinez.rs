//! The Fe–Cr pair-interaction model of Martínez, Senninger, Fu and Soisson,
//! Phys. Rev. B 86, 224109 (2012) (M12), as an [`Energetics`] implementation.
//!
//! Bonds are 1NN and 2NN (2NN = half the 1NN value). Pair free energies depend on
//! the temperature through the Einstein-plus-linear heat capacities of Table I; the
//! Fe–Cr bond additionally depends on the local Cr concentration through Ω(x).
//! Saddle-point interactions g̃ are fitted at construction so that the pure-metal
//! migration free energies and the Cr-solute-in-Fe (Fe-solute-in-Cr) values of
//! Table I are reproduced exactly.
//!
//! Values are transcribed from `tests/data/fecr-martinez2012.json`, a copy of the
//! local parameter file. Test T1 keeps them equal.
//!
//! Interpretation choices:
//! - A1: h_AA = +2H/11, not the printed −2H/11, and likewise s_AA = +2S/11.
//! - A2: s_FeCr carries −Ω/(11Θ).
//! - A3: local x per Levesque, divided by 30.
//! - A4: saddle spectators count both jump ends as non-Cr.
//! - A5: explicit fit of the saddle interactions.
//! - A6: g̃ index order is `[jumper][neighbour]`.
//! - A9: standard Einstein c_P.
//! - A10: g̃ is refitted at the operating temperature.
//!
//! Barrier floor (okrates's choice; neither M12 nor S16 states it): the pair model
//! has no global lower bound on E_saddle − E_initial, and the barrier can fall below
//! the exchange energy ΔE = E_final − E_initial, which would give a negative
//! reverse barrier. Where the pair model puts the saddle free energy below
//! max(G_initial, G_final), [`Energetics::barrier`] raises it to that level, so the
//! returned barrier is max(raw, max(0, ΔE)). The floored saddle level is the same
//! for the forward and the reverse jump, so detailed balance stays exact. Each
//! evaluation in which the floor acts is counted
//! ([`Martinez2012::floor_activations`]). [`Martinez2012::raw_barrier`] gives the
//! unfloored value. A panic from [`Energetics::barrier`] means a non-finite energy.

use crate::bcc::BccLattice;
use crate::fecr::{Energetics, FeCrError, Species};
use crate::vacancy::BOLTZMANN_EV_PER_K;

/// Parameters of the model. Energies in eV, entropies in k_B, temperatures in K.
pub mod params {
    /// Cohesive enthalpy H0 of Fe in eV (M12 Table I, p.4).
    pub const H0_FE: f64 = -4.28;
    /// Cohesive enthalpy H0 of Cr in eV (M12 Table I, p.4).
    pub const H0_CR: f64 = -4.10;
    /// Einstein temperature of the Fe heat capacity in K (M12 Table I, p.4).
    pub const CP_THETA_FE: f64 = 310.0;
    /// Linear term of the Fe heat capacity in k_B/K (M12 Table I, p.4).
    pub const CP_LINEAR_FE: f64 = 6.64e-4;
    /// Einstein temperature of the Cr heat capacity in K (M12 Table I, p.4).
    pub const CP_THETA_CR: f64 = 360.0;
    /// Linear term of the Cr heat capacity in k_B/K (M12 Table I, p.4).
    pub const CP_LINEAR_CR: f64 = 5.0e-4;
    /// Vacancy formation enthalpy in Fe in eV (M12 Table I, p.4).
    pub const DHF_V_FE: f64 = 2.18;
    /// Vacancy formation enthalpy in Cr in eV (M12 Table I, p.4).
    pub const DHF_V_CR: f64 = 2.33;
    /// Vacancy formation entropy in Fe in k_B (M12 Table I, p.4).
    pub const DSF_V_FE: f64 = 4.1;
    /// Vacancy formation entropy in Cr in k_B (M12 Table I, p.4).
    pub const DSF_V_CR: f64 = 4.1;
    /// Temperature Θ in the Fe–Cr mixing entropy, in K (M12 Table I, p.4).
    pub const THETA_K: f64 = 1480.0;
    /// Migration enthalpy of Fe in pure Fe in eV (M12 Table I, p.4).
    pub const DH0_FE: f64 = 0.69;
    /// Migration enthalpy of a Cr solute in Fe in eV (M12 Table I, p.4).
    pub const DH2_FE: f64 = 0.57;
    /// Migration enthalpy of Cr in pure Cr in eV (M12 Table I, p.4).
    pub const DH0_CR: f64 = 0.78;
    /// Migration enthalpy of an Fe solute in Cr in eV (M12 Table I, p.4).
    pub const DH2_CR: f64 = 0.70;
    /// Migration entropy in k_B (M12 Table I, p.4).
    pub const DS_MIG: f64 = 2.1;
    /// Ω(x) coefficient α, PWSCF-PAW (Levesque, arXiv:1307.7561, Table I).
    pub const OMEGA_ALPHA: f64 = 0.070;
    /// Ω(x) coefficient β in eV, PWSCF-PAW (Levesque, arXiv:1307.7561, Table I).
    pub const OMEGA_BETA: f64 = -2.288;
    /// Ω(x) coefficient γ in eV, PWSCF-PAW (Levesque, arXiv:1307.7561, Table I).
    pub const OMEGA_GAMMA: f64 = 4.439;
    /// Ω(x) coefficient δ in eV, PWSCF-PAW (Levesque, arXiv:1307.7561, Table I).
    pub const OMEGA_DELTA: f64 = -2.480;
    /// Attempt frequency of Fe in s⁻¹ (M12 p.6, Sec. II C 4b).
    pub const NU_FE: f64 = 1e13;
    /// Attempt frequency of Cr in s⁻¹ (M12 p.6, Sec. II C 4b).
    pub const NU_CR: f64 = 1e13;
}

use params::*;

/// Number of sites in {s} ∪ 1NN ∪ 2NN, twice of which is the local-x denominator.
const LOCAL_DENOMINATOR: f64 = 30.0;
const MIN_CELLS: usize = 3;

/// A site's content for energy evaluation; `Empty` has no bonds (saddle ends).
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

/// An occupancy with up to two sites overridden.
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

    /// Cr atoms on {s} ∪ 1NN(s) ∪ 2NN(s); vacancies and empty sites are non-Cr.
    ///
    /// Counts on the base occupancy, then corrects for the overridden sites. The
    /// loops are written out because this is the hot path of every local delta.
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

impl Martinez2012 {
    /// Free energy of a 1NN bond between cells `a` and `b`; `n_sum` = n(s) + n(t) is
    /// used only by Fe–Cr bonds.
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

    /// Σ over every bond touching `region` (each once) of its energy in `after` minus in `before`.
    ///
    /// Summing per-bond differences keeps unchanged bonds at exactly zero, so the
    /// result carries no rounding error from the large total energy. `changed` are
    /// the sites whose content differs between the views: a bond that touches none of
    /// them and is not an Fe–Cr bond (the only kind that depends on neighbouring
    /// sites) has an exactly zero difference and is skipped.
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
}

fn omega(x: f64) -> f64 {
    (x - OMEGA_ALPHA) * (OMEGA_BETA * x * x + OMEGA_GAMMA * x + OMEGA_DELTA)
}

/// ∫ c_P dT and ∫ c_P/T dT in eV and eV/K (Einstein plus linear term).
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

/// Sorted union of {s} ∪ 1NN ∪ 2NN over the given sites.
///
/// Insertion sort with explicit loops: at most 30 entries, and cheaper than the
/// generic sort in unoptimised builds.
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

/// The M12 Fe–Cr energetics at a fixed temperature.
#[derive(Clone, Debug)]
pub struct Martinez2012 {
    temperature_k: f64,
    nu: [f64; 2],
    saddle: [[f64; 2]; 2],
    /// 1NN pair free energies g_FeFe, g_CrCr.
    g_aa: [f64; 2],
    /// 1NN pair free energies g_FeV, g_CrV.
    g_av: [f64; 2],
    /// (1 − T/Θ)/11, the factor of Ω in the Fe–Cr bond.
    omega_scale: f64,
    /// Number of barrier evaluations in which the floor acted; shared by clones.
    floor_activations: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl Martinez2012 {
    /// Creates the model at `temperature_k` (finite, ≥ 0; 0 K is allowed for checks)
    /// with the paper's attempt frequencies and fits the saddle interactions.
    pub fn new(temperature_k: f64) -> Result<Self, FeCrError> {
        Self::build(temperature_k, NU_FE, NU_CR)
    }

    /// Like [`new`](Self::new) with explicit attempt frequencies (s⁻¹, finite and > 0).
    ///
    /// Exists for tests and sensitivity studies.
    pub fn with_attempt_frequencies(
        temperature_k: f64,
        nu_fe: f64,
        nu_cr: f64,
    ) -> Result<Self, FeCrError> {
        for nu in [nu_fe, nu_cr] {
            if !nu.is_finite() || nu <= 0.0 {
                return Err(FeCrError::AttemptFrequency(nu));
            }
        }
        Self::build(temperature_k, nu_fe, nu_cr)
    }

    fn build(temperature_k: f64, nu_fe: f64, nu_cr: f64) -> Result<Self, FeCrError> {
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
            floor_activations: std::sync::Arc::default(),
        };
        model.fit_saddle();
        Ok(model)
    }

    /// Fits the saddle interactions on a private 6×6×6 lattice.
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

    /// Saddle interactions g̃ in eV, indexed `[jumper][neighbour]` with Fe = 0, Cr = 1.
    pub fn saddle_interactions(&self) -> [[f64; 2]; 2] {
        self.saddle
    }

    /// Number of [`Energetics::barrier`] evaluations in which the floor raised the
    /// barrier above the pair-model value.
    ///
    /// Clones share the counter, so every replica built from one instance adds to
    /// it. It counts barrier evaluations, not fired jumps.
    pub fn floor_activations(&self) -> u64 {
        self.floor_activations
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Resets the floor-activation counter (shared with all clones) to zero.
    pub fn reset_floor_activations(&self) {
        self.floor_activations
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// E_saddle − E_initial in eV without the floor, for diagnostics and tests.
    pub fn raw_barrier(
        &self,
        lattice: &BccLattice,
        occupancy: &[Species],
        vacancy_site: usize,
        atom_site: usize,
    ) -> f64 {
        self.barrier_raw(lattice, occupancy, vacancy_site, atom_site)
    }

    /// Adds `shift_ev` to all four fitted saddle interactions, keeping the counter.
    ///
    /// Exists only to force the barrier floor in tests.
    #[doc(hidden)]
    pub fn with_saddle_shift(mut self, shift_ev: f64) -> Self {
        for row in &mut self.saddle {
            for g in row {
                *g += shift_ev;
            }
        }
        self
    }

    /// The temperature in K the energetics were built for.
    pub fn temperature_k(&self) -> f64 {
        self.temperature_k
    }

    /// E_after − E_before over the bonds touching the regions of `sites`, with `after` placed on `sites`.
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

    /// Bond free energy between two site contents; `None` is an empty site (no bonds).
    pub(crate) fn bond_energy(&self, a: Option<Species>, b: Option<Species>, n_sum: u32) -> f64 {
        self.bond(
            a.map_or(Cell::Empty, Cell::from),
            b.map_or(Cell::Empty, Cell::from),
            n_sum,
        )
    }

    /// Barrier without the sign check; `atom_site` jumps into the vacancy at `vacancy_site`.
    fn barrier_raw(
        &self,
        lattice: &BccLattice,
        occupancy: &[Species],
        vacancy_site: usize,
        atom_site: usize,
    ) -> f64 {
        let (q, p) = (vacancy_site, atom_site);
        assert_eq!(
            occupancy[q],
            Species::Vacancy,
            "barrier: site {q} is not the vacancy"
        );
        let jumper = match occupancy[p] {
            Species::Fe => 0,
            Species::Cr => 1,
            Species::Vacancy => panic!("barrier: jumping site {p} is a vacancy"),
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
                Species::Vacancy => {
                    panic!(
                        "barrier: saddle neighbour {k} is a vacancy; only single-vacancy states are supported"
                    )
                }
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
        self.local_delta(
            lattice,
            occupancy,
            [vacancy_site, atom_site],
            [occupancy[atom_site].into(), Cell::Vacancy],
        )
    }

    fn swap_delta(&self, lattice: &BccLattice, occupancy: &[Species], i: usize, j: usize) -> f64 {
        self.local_delta(
            lattice,
            occupancy,
            [i, j],
            [occupancy[j].into(), occupancy[i].into()],
        )
    }

    /// Barrier in eV for the atom on `atom_site` jumping into the vacancy, floored
    /// at max(0, ΔE) with ΔE = E_final − E_initial.
    ///
    /// In the saddle state both jump ends are empty (no bonds, non-Cr in local
    /// concentrations) and the six saddle neighbours add g̃[jumper][neighbour].
    ///
    /// The floor is okrates's choice and is not stated in M12 or S16: where the
    /// pair model puts the saddle free energy below max(G_initial, G_final), it is
    /// raised to that level. The floored saddle level is the same for the forward
    /// and the reverse jump, so detailed balance stays exact. Each evaluation in
    /// which the floor acts increments [`Martinez2012::floor_activations`].
    ///
    /// # Panics
    ///
    /// Panics with "non-finite Martinez barrier" if the raw barrier or ΔE is not
    /// finite. Also panics if a saddle neighbour is a vacancy.
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
            panic!(
                "non-finite Martinez barrier {raw} eV, exchange energy {de} eV (vacancy {vacancy_site}, atom {atom_site})"
            );
        }
        let floor = de.max(0.0);
        if raw < floor {
            self.floor_activations
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return floor;
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
