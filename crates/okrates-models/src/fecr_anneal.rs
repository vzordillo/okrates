//! Kawasaki (Fe↔Cr swap) Metropolis equilibration and 1NN Warren–Cowley SRO diagnostics.
//!
//! Spin-exchange kinetics follow Kawasaki, *Phys. Rev.* **145**, 224–230 (1966),
//! doi:10.1103/PhysRev.145.224, with Metropolis acceptance (Metropolis, Rosenbluth, Rosenbluth,
//! Teller, and Teller, *J. Chem. Phys.* **21**, 1087–1092 (1953), doi:10.1063/1.1699114).
//! Composition and vacancy site remain fixed, sampling the canonical Boltzmann
//! distribution conditional on the vacancy position.

use okrates_core::KmcRng;

use crate::bcc::BccLattice;
use crate::fecr::{Energetics, FeCrError, FeCrModel, Species};

/// Warren–Cowley 1NN short-range order `alpha_1 = 1 - P(Cr | Fe) / x_Cr` over atom–atom 1NN bonds
/// (Cowley, *Phys. Rev.* **77**, 669–675 (1950), doi:10.1103/PhysRev.77.669;
/// Warren, *X-Ray Diffraction* (1969)).
///
/// Positive values indicate Fe–Cr clustering/phase separation; negative values indicate ordering.
pub fn warren_cowley_alpha1(lattice: &BccLattice, occupancy: &[Species]) -> Result<f64, FeCrError> {
    let expected = lattice.site_count();
    if occupancy.len() != expected {
        return Err(FeCrError::SiteCount {
            expected,
            found: occupancy.len(),
        });
    }
    let mut n_fe = 0_usize;
    let mut n_cr = 0_usize;
    let mut fe_atom_bonds = 0_usize;
    let mut fe_cr_bonds = 0_usize;

    for (site, &sp) in occupancy.iter().enumerate() {
        match sp {
            Species::Fe => {
                n_fe += 1;
                for &nb in lattice.neighbors(site) {
                    match occupancy[nb] {
                        Species::Fe => fe_atom_bonds += 1,
                        Species::Cr => {
                            fe_atom_bonds += 1;
                            fe_cr_bonds += 1;
                        }
                        Species::Vacancy => {}
                    }
                }
            }
            Species::Cr => n_cr += 1,
            Species::Vacancy => {}
        }
    }

    if n_fe == 0 || n_cr == 0 || fe_atom_bonds == 0 {
        return Ok(0.0);
    }
    let x_cr = n_cr as f64 / (n_fe + n_cr) as f64;
    let p_cr_given_fe = fe_cr_bonds as f64 / fe_atom_bonds as f64;
    Ok(1.0 - p_cr_given_fe / x_cr)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnnealDiagnostics {
    pub acceptance_rate: f64,
    pub final_sro_alpha1: f64,
    /// Mean energy of the final quarter of sweeps minus the preceding quarter in eV (`0.0` if `< 4` sweeps).
    pub tail_energy_drift_ev: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Anneal {
    pub occupancy: Vec<Species>,
    pub attempts: u64,
    pub accepted: u64,
    /// Entry `0` is initial energy; entry `s` is energy after sweep `s` in eV.
    pub energy: Vec<f64>,
    /// Entry `0` is initial `alpha_1`; entry `s` is `alpha_1` after sweep `s`.
    pub sro_alpha1: Vec<f64>,
    pub diagnostics: AnnealDiagnostics,
}

// Møller–Knuth TwoSum (Møller, BIT 5, 37–50 (1965); Knuth, TAOCP Vol. 2, 3rd ed. (1997), §4.2.2, Theorem B).
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    (s, (a - (s - v)) + (b - v))
}

/// Stateful Kawasaki Metropolis annealer preserving RNG and lattice state across sweep chunks.
#[derive(Clone, Debug)]
pub struct KawasakiAnnealer<'a, E: Energetics> {
    model: &'a FeCrModel<E>,
    occupancy: Vec<Species>,
    fe: Vec<usize>,
    cr: Vec<usize>,
    current_energy: f64,
    energy_comp: f64,
    attempts: u64,
    accepted: u64,
    energy: Vec<f64>,
    sro_alpha1: Vec<f64>,
    rng: KmcRng,
}

impl<'a, E: Energetics> KawasakiAnnealer<'a, E> {
    pub fn new(
        model: &'a FeCrModel<E>,
        occupancy: Vec<Species>,
        rng: KmcRng,
    ) -> Result<Self, FeCrError> {
        if !model.energetics().supports_equilibrium_swaps() {
            return Err(FeCrError::UnsupportedEquilibration(
                "tabulated barrier sets define local jump rates only, not global swap energies",
            ));
        }
        model.state(occupancy.clone())?;
        let mut fe = Vec::new();
        let mut cr = Vec::new();
        for (site, &species) in occupancy.iter().enumerate() {
            match species {
                Species::Fe => fe.push(site),
                Species::Cr => cr.push(site),
                Species::Vacancy => {}
            }
        }
        let current_energy = model.energetics().total_energy(model.lattice(), &occupancy);
        let initial_sro = warren_cowley_alpha1(model.lattice(), &occupancy)?;
        Ok(Self {
            model,
            occupancy,
            fe,
            cr,
            current_energy,
            energy_comp: 0.0,
            attempts: 0,
            accepted: 0,
            energy: vec![current_energy],
            sro_alpha1: vec![initial_sro],
            rng,
        })
    }

    pub fn run_sweeps(&mut self, sweeps: u64) -> Result<(), FeCrError> {
        let additional = usize::try_from(sweeps).map_err(|_| FeCrError::SweepAllocation(sweeps))?;
        self.energy
            .try_reserve(additional)
            .map_err(|_| FeCrError::SweepAllocation(sweeps))?;
        self.sro_alpha1
            .try_reserve(additional)
            .map_err(|_| FeCrError::SweepAllocation(sweeps))?;

        let lattice = self.model.lattice();
        let energetics = self.model.energetics();
        let kt = self.model.kt();
        let proposals = if self.fe.is_empty() || self.cr.is_empty() {
            0
        } else {
            self.fe.len() + self.cr.len()
        };
        for _ in 0..sweeps {
            let mut changed = false;
            for _ in 0..proposals {
                let a =
                    ((self.rng.open01() * self.fe.len() as f64) as usize).min(self.fe.len() - 1);
                let b =
                    ((self.rng.open01() * self.cr.len() as f64) as usize).min(self.cr.len() - 1);
                let u = self.rng.open01();
                let delta = energetics.swap_delta(lattice, &self.occupancy, self.fe[a], self.cr[b]);
                self.attempts += 1;
                if u < (-delta / kt).exp() {
                    self.occupancy.swap(self.fe[a], self.cr[b]);
                    std::mem::swap(&mut self.fe[a], &mut self.cr[b]);
                    let prev = self.current_energy;
                    let (s, err) = two_sum(prev, delta);
                    let (hi, lo) = two_sum(s, self.energy_comp + err);
                    if hi.abs() < 1e-3 * prev.abs() {
                        self.current_energy = energetics.total_energy(lattice, &self.occupancy);
                        self.energy_comp = 0.0;
                    } else {
                        self.current_energy = hi;
                        self.energy_comp = lo;
                    }
                    self.accepted += 1;
                    changed = true;
                }
            }
            self.energy.push(self.current_energy);
            let sro = if changed {
                warren_cowley_alpha1(lattice, &self.occupancy)?
            } else {
                self.current_sro_alpha1()
            };
            self.sro_alpha1.push(sro);
        }
        Ok(())
    }

    pub fn occupancy(&self) -> &[Species] {
        &self.occupancy
    }

    pub fn sweeps_completed(&self) -> u64 {
        (self.energy.len() - 1) as u64
    }

    pub fn current_energy(&self) -> f64 {
        self.current_energy
    }

    pub fn current_sro_alpha1(&self) -> f64 {
        *self.sro_alpha1.last().unwrap_or(&0.0)
    }

    pub fn finish(self) -> Anneal {
        let acceptance_rate = if self.attempts == 0 {
            0.0
        } else {
            self.accepted as f64 / self.attempts as f64
        };
        let final_sro_alpha1 = *self.sro_alpha1.last().unwrap_or(&0.0);
        let sweeps = self.energy.len() - 1;
        let tail_energy_drift_ev = if sweeps >= 4 {
            let q = sweeps / 4;
            let prev = &self.energy[sweeps - 2 * q + 1..=sweeps - q];
            let last = &self.energy[sweeps - q + 1..=sweeps];
            let mean_prev = prev.iter().sum::<f64>() / prev.len() as f64;
            let mean_last = last.iter().sum::<f64>() / last.len() as f64;
            mean_last - mean_prev
        } else {
            0.0
        };
        Anneal {
            occupancy: self.occupancy,
            attempts: self.attempts,
            accepted: self.accepted,
            energy: self.energy,
            sro_alpha1: self.sro_alpha1,
            diagnostics: AnnealDiagnostics {
                acceptance_rate,
                final_sro_alpha1,
                tail_energy_drift_ev,
            },
        }
    }
}

/// Runs `sweeps` Kawasaki Metropolis sweeps (`N_Fe + N_Cr` proposals per sweep) from `seed`.
pub fn anneal<E: Energetics>(
    model: &FeCrModel<E>,
    occupancy: Vec<Species>,
    sweeps: u64,
    seed: u64,
) -> Result<Anneal, FeCrError> {
    let mut annealer = KawasakiAnnealer::new(model, occupancy, KmcRng::from_seed(seed))?;
    annealer.run_sweeps(sweeps)?;
    Ok(annealer.finish())
}
