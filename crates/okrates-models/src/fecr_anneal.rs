//! Kawasaki (Fe–Cr swap) Metropolis anneal to equilibrium for the Fe–Cr model's energetics.
//!
//! The vacancy stays fixed and composition is conserved. The chain samples the
//! Boltzmann distribution conditional on the vacancy site, which makes it a
//! way to prepare equilibrated starting configurations.

use okrates_core::KmcRng;

use crate::fecr::{Energetics, FeCrError, FeCrModel, Species};

/// The result of an [`anneal`].
#[derive(Clone, Debug, PartialEq)]
pub struct Anneal {
    /// The occupancy after the last sweep.
    pub occupancy: Vec<Species>,
    /// The number of proposed swaps.
    pub attempts: u64,
    /// The number of accepted swaps.
    pub accepted: u64,
    /// The configuration energy in eV: entry `s` is the energy after sweep `s`, entry 0 the initial energy.
    pub energy: Vec<f64>,
}

/// Anneals `occupancy` with Metropolis Fe↔Cr swaps for `sweeps` sweeps.
///
/// A sweep is as many proposals as there are Fe and Cr atoms. Each proposal
/// picks a uniformly random Fe and a uniformly random Cr and swaps them with
/// probability min(1, exp(−ΔE / k_B T)). Without both species there are no
/// proposals. Fails if `occupancy` is not a valid state of `model`.
pub fn anneal<E: Energetics>(
    model: &FeCrModel<E>,
    occupancy: Vec<Species>,
    sweeps: u64,
    seed: u64,
) -> Result<Anneal, FeCrError> {
    model.state(occupancy.clone())?;
    let lattice = model.lattice();
    let energetics = model.energetics();
    let mut occ = occupancy;
    let mut fe: Vec<usize> = Vec::new();
    let mut cr: Vec<usize> = Vec::new();
    for (site, &species) in occ.iter().enumerate() {
        match species {
            Species::Fe => fe.push(site),
            Species::Cr => cr.push(site),
            Species::Vacancy => {}
        }
    }
    let mut current = energetics.total_energy(lattice, &occ);
    let mut energy = Vec::with_capacity(sweeps as usize + 1);
    energy.push(current);
    let mut attempts = 0;
    let mut accepted = 0;
    let mut rng = KmcRng::from_seed(seed);
    let kt = model.kt();
    let proposals = if fe.is_empty() || cr.is_empty() {
        0
    } else {
        fe.len() + cr.len()
    };
    for _ in 0..sweeps {
        for _ in 0..proposals {
            let a = ((rng.open01() * fe.len() as f64) as usize).min(fe.len() - 1);
            let b = ((rng.open01() * cr.len() as f64) as usize).min(cr.len() - 1);
            let u = rng.open01();
            let delta = energetics.swap_delta(lattice, &occ, fe[a], cr[b]);
            attempts += 1;
            if u < (-delta / kt).exp() {
                occ.swap(fe[a], cr[b]);
                std::mem::swap(&mut fe[a], &mut cr[b]);
                current += delta;
                accepted += 1;
            }
        }
        energy.push(current);
    }
    Ok(Anneal {
        occupancy: occ,
        attempts,
        accepted,
        energy,
    })
}
