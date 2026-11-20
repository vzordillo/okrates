//! Periodic crystal geometry and multi-ion site-exclusion lattice-hopping KMC template.
//!
//! Hopping barriers use the symmetric kinetic-resolved activation (KRA) form
//! `max(E_mig + (E_to - E_from) / 2, 0, E_to - E_from)` with Arrhenius rates
//! `nu * exp(-E_act / (k_B T))`, preserving detailed balance wrt Boltzmann site energies
//! (`BOLTZMANN_EV_PER_K = 1.380649e-23 / 1.602176634e-19 eV/K`).

use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use okrates_core::{
    Checkpointable, IncrementalModel, IndependentSnapshot, KmcRng, Model, ReversibleModel, SumTree,
};

use crate::vacancy::BOLTZMANN_EV_PER_K;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LatticeHoppingError {
    InvalidCells {
        cells: [usize; 3],
        min_cells: [usize; 3],
    },
    InvalidSubTicks([usize; 3]),
    LatticeTooLarge {
        cells: [usize; 3],
    },
    InvalidMetric,
    EmptyBasis,
    BasisOffsetOutOfRange {
        basis: usize,
        offset: [usize; 3],
        sub_ticks: [usize; 3],
    },
    DuplicateBasisOffset([usize; 3]),
    EmptyStencils,
    StencilBasisOutOfRange {
        basis: usize,
        basis_count: usize,
    },
    SelfLoopStencil {
        basis: usize,
        delta_ticks: [i64; 3],
    },
    UnmatchedStencilTarget {
        from_basis: usize,
        delta_ticks: [i64; 3],
    },
    DuplicateDirectedHop {
        site: usize,
        target: usize,
        delta_ticks: [i64; 3],
    },
    MissingReciprocalHop {
        site: usize,
        target: usize,
        delta_ticks: [i64; 3],
    },
    SiteTypeOutOfRange {
        site: usize,
        site_type: usize,
        type_count: usize,
    },
    PathwayOutOfRange {
        pathway: usize,
        pathway_count: usize,
    },
    InvalidSiteEnergy {
        site_type: usize,
        energy_ev: f64,
    },
    InvalidTemperature(f64),
    InvalidAttemptFrequency {
        pathway: usize,
        frequency_hz: f64,
    },
    InvalidMigrationBarrier {
        pathway: usize,
        barrier_ev: f64,
    },
    InvalidJumpRate(f64),
    ZeroIons,
    TooManyIons {
        ions: usize,
        sites: usize,
    },
    TooManySlots(usize),
    SiteOutOfRange {
        site: usize,
        sites: usize,
    },
    DuplicateIonSite(usize),
    IonOutOfRange {
        ion: usize,
        ions: usize,
    },
    BranchOutOfRange {
        ion: usize,
        branch: usize,
        degree: usize,
    },
    TargetOccupied {
        ion: usize,
        target: usize,
    },
    DisplacementOverflow {
        ion: usize,
    },
    HopCountOverflow {
        ion: usize,
    },
    StateMismatch(&'static str),
}

impl fmt::Display for LatticeHoppingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCells { cells, min_cells } => write!(
                f,
                "invalid cell counts {cells:?}; minimum per axis is {min_cells:?}"
            ),
            Self::InvalidSubTicks(sub_ticks) => {
                write!(f, "sub-cell ticks {sub_ticks:?} must all be >= 1")
            }
            Self::LatticeTooLarge { cells } => {
                write!(f, "lattice with cells {cells:?} exceeds capacity limits")
            }
            Self::InvalidMetric => write!(f, "tick metric vectors must be finite and independent"),
            Self::EmptyBasis => write!(f, "unit cell must contain at least one basis site"),
            Self::BasisOffsetOutOfRange {
                basis,
                offset,
                sub_ticks,
            } => write!(
                f,
                "basis site {basis} offset {offset:?} outside sub-ticks {sub_ticks:?}"
            ),
            Self::DuplicateBasisOffset(offset) => {
                write!(f, "duplicate basis offset {offset:?} in unit cell")
            }
            Self::EmptyStencils => write!(f, "lattice requires at least one hop stencil"),
            Self::StencilBasisOutOfRange { basis, basis_count } => write!(
                f,
                "hop stencil origin basis {basis} >= basis count {basis_count}"
            ),
            Self::SelfLoopStencil { basis, delta_ticks } => {
                write!(
                    f,
                    "stencil at basis {basis} with delta {delta_ticks:?} forms a self-loop"
                )
            }
            Self::UnmatchedStencilTarget {
                from_basis,
                delta_ticks,
            } => write!(
                f,
                "stencil from basis {from_basis} by {delta_ticks:?} does not land on any basis site"
            ),
            Self::DuplicateDirectedHop {
                site,
                target,
                delta_ticks,
            } => write!(
                f,
                "duplicate directed hop from site {site} to {target} with delta {delta_ticks:?}"
            ),
            Self::MissingReciprocalHop {
                site,
                target,
                delta_ticks,
            } => write!(
                f,
                "hop from site {site} to {target} (delta {delta_ticks:?}) lacks a matching reciprocal hop"
            ),
            Self::SiteTypeOutOfRange {
                site,
                site_type,
                type_count,
            } => write!(
                f,
                "site {site} has site_type {site_type} outside {type_count} energy levels"
            ),
            Self::PathwayOutOfRange {
                pathway,
                pathway_count,
            } => write!(
                f,
                "hop references pathway {pathway} outside {pathway_count} defined pathways"
            ),
            Self::InvalidSiteEnergy {
                site_type,
                energy_ev,
            } => write!(
                f,
                "site energy for type {site_type} must be finite, got {energy_ev}"
            ),
            Self::InvalidTemperature(t) => write!(f, "temperature {t} K must be finite and > 0"),
            Self::InvalidAttemptFrequency {
                pathway,
                frequency_hz,
            } => write!(
                f,
                "pathway {pathway} attempt frequency {frequency_hz} Hz must be finite and > 0"
            ),
            Self::InvalidMigrationBarrier {
                pathway,
                barrier_ev,
            } => write!(
                f,
                "pathway {pathway} migration barrier {barrier_ev} eV must be finite and >= 0"
            ),
            Self::InvalidJumpRate(r) => write!(f, "jump rate {r} must be finite and >= 0"),
            Self::ZeroIons => write!(f, "at least one mobile ion is required"),
            Self::TooManyIons { ions, sites } => {
                write!(f, "{ions} ions exceed {sites} available lattice sites")
            }
            Self::TooManySlots(slots) => write!(f, "{slots} slots exceed SumTree capacity"),
            Self::SiteOutOfRange { site, sites } => {
                write!(f, "site index {site} out of range for {sites} sites")
            }
            Self::DuplicateIonSite(site) => write!(f, "site {site} occupied by multiple ions"),
            Self::IonOutOfRange { ion, ions } => {
                write!(f, "ion {ion} out of range for {ions} ions")
            }
            Self::BranchOutOfRange {
                ion,
                branch,
                degree,
            } => write!(
                f,
                "ion {ion} branch {branch} out of range for site degree {degree}"
            ),
            Self::TargetOccupied { ion, target } => {
                write!(f, "ion {ion} target site {target} is occupied")
            }
            Self::DisplacementOverflow { ion } => {
                write!(
                    f,
                    "unwrapped tick displacement for ion {ion} overflowed i64"
                )
            }
            Self::HopCountOverflow { ion } => write!(f, "hop counter for ion {ion} overflowed u64"),
            Self::StateMismatch(msg) => {
                write!(f, "lattice hopping state invariant violated: {msg}")
            }
        }
    }
}

impl Error for LatticeHoppingError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectedHop {
    pub target: usize,
    pub delta_ticks: [i64; 3],
    pub pathway: usize,
    pub reverse_branch: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BasisSite {
    pub offset_ticks: [usize; 3],
    pub site_type: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HopStencil {
    pub from_basis: usize,
    pub delta_ticks: [i64; 3],
    pub pathway: usize,
}

/// Periodic or partially confined Bravais/multi-sublattice crystal geometry on an integer tick grid.
#[derive(Clone, Debug, PartialEq)]
pub struct PeriodicLattice {
    cells: [usize; 3],
    sub_ticks: [usize; 3],
    period_ticks: [i64; 3],
    periodic_axes: [bool; 3],
    spatial_dimension: usize,
    tick_vectors_m: [[f64; 3]; 3],
    site_coords_ticks: Vec<[i64; 3]>,
    site_types: Vec<usize>,
    adjacency: Vec<Vec<DirectedHop>>,
    max_degree: usize,
    type_count: usize,
    pathway_count: usize,
}

fn vec3_norm_sq(v: [f64; 3]) -> f64 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

fn normalize_axis(v: [f64; 3]) -> Option<[f64; 3]> {
    let scale = v[0].abs().max(v[1].abs()).max(v[2].abs());
    if !scale.is_finite() || scale == 0.0 {
        return None;
    }
    Some([v[0] / scale, v[1] / scale, v[2] / scale])
}

fn validate_metric(
    tick_vectors_m: [[f64; 3]; 3],
    active_axes: [bool; 3],
) -> Result<usize, LatticeHoppingError> {
    for row in &tick_vectors_m {
        if row.iter().any(|&v| !v.is_finite()) {
            return Err(LatticeHoppingError::InvalidMetric);
        }
    }
    let active: Vec<usize> = (0..3).filter(|&a| active_axes[a]).collect();
    let ok = match active.as_slice() {
        [] => false,
        [a] => normalize_axis(tick_vectors_m[*a]).is_some(),
        [a, b] => match (
            normalize_axis(tick_vectors_m[*a]),
            normalize_axis(tick_vectors_m[*b]),
        ) {
            (Some(u), Some(v)) => {
                let cross = [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ];
                vec3_norm_sq(cross) > 0.0
            }
            _ => false,
        },
        [0, 1, 2] => match (
            normalize_axis(tick_vectors_m[0]),
            normalize_axis(tick_vectors_m[1]),
            normalize_axis(tick_vectors_m[2]),
        ) {
            (Some(a), Some(b), Some(c)) => {
                let det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0]);
                det != 0.0 && det.is_finite()
            }
            _ => false,
        },
        _ => unreachable!(),
    };
    if !ok {
        return Err(LatticeHoppingError::InvalidMetric);
    }
    Ok(active.len())
}

impl PeriodicLattice {
    /// Builds a crystal from integer cell dimensions, sub-cell tick subdivisions, boundary flags,
    /// physical tick basis vectors (in meters), unit-cell basis sites, and reciprocal hop stencils.
    pub fn from_unit_cell(
        cells: [usize; 3],
        sub_ticks: [usize; 3],
        periodic_axes: [bool; 3],
        tick_vectors_m: [[f64; 3]; 3],
        basis: &[BasisSite],
        stencils: &[HopStencil],
    ) -> Result<Self, LatticeHoppingError> {
        let min_cells = std::array::from_fn(|a| if periodic_axes[a] { 2 } else { 1 });
        if (0..3).any(|a| cells[a] < min_cells[a]) {
            return Err(LatticeHoppingError::InvalidCells { cells, min_cells });
        }
        if sub_ticks.contains(&0) {
            return Err(LatticeHoppingError::InvalidSubTicks(sub_ticks));
        }
        if basis.is_empty() {
            return Err(LatticeHoppingError::EmptyBasis);
        }
        if stencils.is_empty() {
            return Err(LatticeHoppingError::EmptyStencils);
        }

        let cell_count = cells[0]
            .checked_mul(cells[1])
            .and_then(|n| n.checked_mul(cells[2]))
            .ok_or(LatticeHoppingError::LatticeTooLarge { cells })?;
        let site_count = cell_count
            .checked_mul(basis.len())
            .filter(|&n| u32::try_from(n).is_ok())
            .ok_or(LatticeHoppingError::LatticeTooLarge { cells })?;
        let period_ticks: [Option<i64>; 3] = std::array::from_fn(|a| {
            cells[a]
                .checked_mul(sub_ticks[a])
                .and_then(|n| i64::try_from(n).ok())
        });
        let period_ticks = [
            period_ticks[0].ok_or(LatticeHoppingError::LatticeTooLarge { cells })?,
            period_ticks[1].ok_or(LatticeHoppingError::LatticeTooLarge { cells })?,
            period_ticks[2].ok_or(LatticeHoppingError::LatticeTooLarge { cells })?,
        ];

        let active_axes = std::array::from_fn(|a| {
            periodic_axes[a]
                || cells[a] > 1
                || basis.iter().any(|b| b.offset_ticks[a] > 0)
                || stencils.iter().any(|s| s.delta_ticks[a] != 0)
        });
        let spatial_dimension = validate_metric(tick_vectors_m, active_axes)?;

        let mut basis_map = HashMap::with_capacity(basis.len());
        let mut type_count = 0_usize;
        for (b_idx, b) in basis.iter().enumerate() {
            for a in 0..3 {
                if b.offset_ticks[a] >= sub_ticks[a] {
                    return Err(LatticeHoppingError::BasisOffsetOutOfRange {
                        basis: b_idx,
                        offset: b.offset_ticks,
                        sub_ticks,
                    });
                }
            }
            if basis_map.insert(b.offset_ticks, b_idx).is_some() {
                return Err(LatticeHoppingError::DuplicateBasisOffset(b.offset_ticks));
            }
            let next_type =
                b.site_type
                    .checked_add(1)
                    .ok_or(LatticeHoppingError::SiteTypeOutOfRange {
                        site: 0,
                        site_type: b.site_type,
                        type_count: usize::MAX,
                    })?;
            type_count = type_count.max(next_type);
        }

        let mut per_basis_stencils: Vec<Vec<(HopStencil, usize, [i128; 3])>> =
            vec![Vec::new(); basis.len()];
        let mut pathway_count = 0_usize;
        for s in stencils {
            if s.from_basis >= basis.len() {
                return Err(LatticeHoppingError::StencilBasisOutOfRange {
                    basis: s.from_basis,
                    basis_count: basis.len(),
                });
            }
            if s.delta_ticks == [0, 0, 0] {
                return Err(LatticeHoppingError::SelfLoopStencil {
                    basis: s.from_basis,
                    delta_ticks: s.delta_ticks,
                });
            }
            let next_pw =
                s.pathway
                    .checked_add(1)
                    .ok_or(LatticeHoppingError::PathwayOutOfRange {
                        pathway: s.pathway,
                        pathway_count: usize::MAX,
                    })?;
            pathway_count = pathway_count.max(next_pw);

            let orig = basis[s.from_basis].offset_ticks;
            let mut target_offset = [0_usize; 3];
            let mut cell_shift = [0_i128; 3];
            for a in 0..3 {
                let sum = i128::from(orig[a] as i64) + i128::from(s.delta_ticks[a]);
                let sub = i128::from(sub_ticks[a] as i64);
                target_offset[a] = sum.rem_euclid(sub) as usize;
                cell_shift[a] = sum.div_euclid(sub);
            }
            let &target_basis = basis_map.get(&target_offset).ok_or(
                LatticeHoppingError::UnmatchedStencilTarget {
                    from_basis: s.from_basis,
                    delta_ticks: s.delta_ticks,
                },
            )?;
            per_basis_stencils[s.from_basis].push((*s, target_basis, cell_shift));
        }

        let mut site_coords_ticks = Vec::new();
        let mut site_types = Vec::new();
        let mut adjacency = Vec::new();
        if site_coords_ticks.try_reserve_exact(site_count).is_err()
            || site_types.try_reserve_exact(site_count).is_err()
            || adjacency.try_reserve_exact(site_count).is_err()
        {
            return Err(LatticeHoppingError::LatticeTooLarge { cells });
        }

        let basis_len = basis.len();
        let site_index = |cx: usize, cy: usize, cz: usize, b: usize| -> usize {
            (((cx * cells[1] + cy) * cells[2] + cz) * basis_len) + b
        };

        for cx in 0..cells[0] {
            for cy in 0..cells[1] {
                for cz in 0..cells[2] {
                    let c = [cx, cy, cz];
                    for (b_idx, b) in basis.iter().enumerate() {
                        let site = site_coords_ticks.len();
                        let q = std::array::from_fn(|a| {
                            (c[a] * sub_ticks[a] + b.offset_ticks[a]) as i64
                        });
                        site_coords_ticks.push(q);
                        site_types.push(b.site_type);

                        let mut hops = Vec::with_capacity(per_basis_stencils[b_idx].len());
                        for &(st, target_basis, shift) in &per_basis_stencils[b_idx] {
                            let mut tc = [0_usize; 3];
                            let mut in_bounds = true;
                            for a in 0..3 {
                                let raw = (c[a] as i128) + shift[a];
                                if periodic_axes[a] {
                                    tc[a] = raw.rem_euclid(cells[a] as i128) as usize;
                                } else if raw < 0 || raw >= (cells[a] as i128) {
                                    in_bounds = false;
                                    break;
                                } else {
                                    tc[a] = raw as usize;
                                }
                            }
                            if !in_bounds {
                                continue;
                            }
                            let target = site_index(tc[0], tc[1], tc[2], target_basis);
                            if target == site {
                                return Err(LatticeHoppingError::SelfLoopStencil {
                                    basis: b_idx,
                                    delta_ticks: st.delta_ticks,
                                });
                            }
                            if hops.iter().any(|h: &DirectedHop| {
                                h.target == target && h.delta_ticks == st.delta_ticks
                            }) {
                                return Err(LatticeHoppingError::DuplicateDirectedHop {
                                    site,
                                    target,
                                    delta_ticks: st.delta_ticks,
                                });
                            }
                            hops.push(DirectedHop {
                                target,
                                delta_ticks: st.delta_ticks,
                                pathway: st.pathway,
                                reverse_branch: 0,
                            });
                        }
                        adjacency.push(hops);
                    }
                }
            }
        }

        let mut max_degree = 0_usize;
        for site in 0..site_count {
            max_degree = max_degree.max(adjacency[site].len());
            let deg = adjacency[site].len();
            for branch in 0..deg {
                let hop = adjacency[site][branch];
                let neg_delta = [
                    hop.delta_ticks[0].checked_neg(),
                    hop.delta_ticks[1].checked_neg(),
                    hop.delta_ticks[2].checked_neg(),
                ];
                let (Some(nx), Some(ny), Some(nz)) = (neg_delta[0], neg_delta[1], neg_delta[2])
                else {
                    return Err(LatticeHoppingError::MissingReciprocalHop {
                        site,
                        target: hop.target,
                        delta_ticks: hop.delta_ticks,
                    });
                };
                let wanted = [nx, ny, nz];
                let rev_branch = adjacency[hop.target]
                    .iter()
                    .position(|rh| {
                        rh.target == site && rh.delta_ticks == wanted && rh.pathway == hop.pathway
                    })
                    .ok_or(LatticeHoppingError::MissingReciprocalHop {
                        site,
                        target: hop.target,
                        delta_ticks: hop.delta_ticks,
                    })?;
                adjacency[site][branch].reverse_branch = rev_branch;
            }
        }

        Ok(Self {
            cells,
            sub_ticks,
            period_ticks,
            periodic_axes,
            spatial_dimension,
            tick_vectors_m,
            site_coords_ticks,
            site_types,
            adjacency,
            max_degree,
            type_count,
            pathway_count,
        })
    }

    /// Periodic 3D simple cubic lattice (`z = 6`, jump length `lattice_a_m`).
    pub fn simple_cubic(cells: [usize; 3], lattice_a_m: f64) -> Result<Self, LatticeHoppingError> {
        if !lattice_a_m.is_finite() || lattice_a_m <= 0.0 {
            return Err(LatticeHoppingError::InvalidMetric);
        }
        let basis = [BasisSite {
            offset_ticks: [0, 0, 0],
            site_type: 0,
        }];
        let deltas = [
            [-1, 0, 0],
            [1, 0, 0],
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, -1],
            [0, 0, 1],
        ];
        let stencils: Vec<HopStencil> = deltas
            .into_iter()
            .map(|delta_ticks| HopStencil {
                from_basis: 0,
                delta_ticks,
                pathway: 0,
            })
            .collect();
        let metric = [
            [lattice_a_m, 0.0, 0.0],
            [0.0, lattice_a_m, 0.0],
            [0.0, 0.0, lattice_a_m],
        ];
        Self::from_unit_cell(
            cells,
            [1, 1, 1],
            [true, true, true],
            metric,
            &basis,
            &stencils,
        )
    }

    /// Periodic 3D face-centered cubic lattice (`z = 12`, conventional cell `lattice_a_m`, jump length `a / sqrt(2)`).
    pub fn fcc(cells: [usize; 3], lattice_a_m: f64) -> Result<Self, LatticeHoppingError> {
        if !lattice_a_m.is_finite() || lattice_a_m <= 0.0 {
            return Err(LatticeHoppingError::InvalidMetric);
        }
        let basis = [
            BasisSite {
                offset_ticks: [0, 0, 0],
                site_type: 0,
            },
            BasisSite {
                offset_ticks: [0, 1, 1],
                site_type: 0,
            },
            BasisSite {
                offset_ticks: [1, 0, 1],
                site_type: 0,
            },
            BasisSite {
                offset_ticks: [1, 1, 0],
                site_type: 0,
            },
        ];
        let nn_deltas = [
            [1, 1, 0],
            [1, -1, 0],
            [-1, 1, 0],
            [-1, -1, 0],
            [1, 0, 1],
            [1, 0, -1],
            [-1, 0, 1],
            [-1, 0, -1],
            [0, 1, 1],
            [0, 1, -1],
            [0, -1, 1],
            [0, -1, -1],
        ];
        let mut stencils = Vec::with_capacity(basis.len() * nn_deltas.len());
        for from_basis in 0..basis.len() {
            for &delta_ticks in &nn_deltas {
                stencils.push(HopStencil {
                    from_basis,
                    delta_ticks,
                    pathway: 0,
                });
            }
        }
        let half_a = 0.5 * lattice_a_m;
        let metric = [[half_a, 0.0, 0.0], [0.0, half_a, 0.0], [0.0, 0.0, half_a]];
        Self::from_unit_cell(
            cells,
            [2, 2, 2],
            [true, true, true],
            metric,
            &basis,
            &stencils,
        )
    }

    /// Periodic 2D square lattice (`z = 4`, jump length `lattice_a_m`, `z` displacement identically zero).
    pub fn square_2d(cells: [usize; 2], lattice_a_m: f64) -> Result<Self, LatticeHoppingError> {
        if !lattice_a_m.is_finite() || lattice_a_m <= 0.0 {
            return Err(LatticeHoppingError::InvalidMetric);
        }
        let basis = [BasisSite {
            offset_ticks: [0, 0, 0],
            site_type: 0,
        }];
        let stencils = [
            HopStencil {
                from_basis: 0,
                delta_ticks: [-1, 0, 0],
                pathway: 0,
            },
            HopStencil {
                from_basis: 0,
                delta_ticks: [1, 0, 0],
                pathway: 0,
            },
            HopStencil {
                from_basis: 0,
                delta_ticks: [0, -1, 0],
                pathway: 0,
            },
            HopStencil {
                from_basis: 0,
                delta_ticks: [0, 1, 0],
                pathway: 0,
            },
        ];
        let metric = [
            [lattice_a_m, 0.0, 0.0],
            [0.0, lattice_a_m, 0.0],
            [0.0, 0.0, 0.0],
        ];
        Self::from_unit_cell(
            [cells[0], cells[1], 1],
            [1, 1, 1],
            [true, true, false],
            metric,
            &basis,
            &stencils,
        )
    }

    pub fn cells(&self) -> [usize; 3] {
        self.cells
    }

    pub fn sub_ticks(&self) -> [usize; 3] {
        self.sub_ticks
    }

    pub fn period_ticks(&self) -> [i64; 3] {
        self.period_ticks
    }

    pub fn periodic_axes(&self) -> [bool; 3] {
        self.periodic_axes
    }

    pub fn spatial_dimension(&self) -> usize {
        self.spatial_dimension
    }

    pub fn tick_vectors_m(&self) -> [[f64; 3]; 3] {
        self.tick_vectors_m
    }

    pub fn site_count(&self) -> usize {
        self.site_coords_ticks.len()
    }

    pub fn max_degree(&self) -> usize {
        self.max_degree
    }

    pub fn type_count(&self) -> usize {
        self.type_count
    }

    pub fn pathway_count(&self) -> usize {
        self.pathway_count
    }

    pub fn degree(&self, site: usize) -> usize {
        self.adjacency.get(site).map_or(0, Vec::len)
    }

    pub fn hops(&self, site: usize) -> &[DirectedHop] {
        self.adjacency.get(site).map_or(&[], Vec::as_slice)
    }

    pub fn site_type(&self, site: usize) -> Option<usize> {
        self.site_types.get(site).copied()
    }

    pub fn site_coord_ticks(&self, site: usize) -> Option<[i64; 3]> {
        self.site_coords_ticks.get(site).copied()
    }

    pub fn displacement_m(&self, delta_ticks: [i64; 3]) -> [f64; 3] {
        let [v0, v1, v2] = self.tick_vectors_m;
        let dx = delta_ticks[0] as f64;
        let dy = delta_ticks[1] as f64;
        let dz = delta_ticks[2] as f64;
        [
            v0[0] * dx + v1[0] * dy + v2[0] * dz,
            v0[1] * dx + v1[1] * dy + v2[1] * dz,
            v0[2] * dx + v1[2] * dy + v2[2] * dz,
        ]
    }

    pub fn squared_displacement_m2(&self, delta_ticks: [i64; 3]) -> f64 {
        vec3_norm_sq(self.displacement_m(delta_ticks))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathwayParams {
    pub attempt_frequency_hz: f64,
    pub migration_barrier_ev: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IonHop {
    pub ion: usize,
    pub branch: usize,
}

/// Allocation-free single-use rollback token for [`ReversibleModel::undo`].
///
/// Must be applied once directly to the state produced by its matching
/// [`ReversibleModel::apply_reversible`] call before any other mutation.
#[derive(Debug, PartialEq, Eq)]
pub struct IonHopUndo {
    ion: usize,
    from_site: usize,
    to_site: usize,
    prev_disp: [i64; 3],
    prev_last_moved_from: Option<(usize, usize)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LatticeHoppingState {
    site_ion: Vec<Option<usize>>,
    sites: Vec<usize>,
    origins: Vec<usize>,
    displacements_ticks: Vec<[i64; 3]>,
    hop_counts: Vec<u64>,
    last_moved_from: Option<(usize, usize)>,
}

impl IndependentSnapshot for LatticeHoppingState {
    fn snapshot(&self) -> Self {
        self.clone()
    }
}

impl LatticeHoppingState {
    pub fn ion_count(&self) -> usize {
        self.sites.len()
    }

    pub fn sites(&self) -> &[usize] {
        &self.sites
    }

    pub fn origins(&self) -> &[usize] {
        &self.origins
    }

    pub fn site_ion(&self) -> &[Option<usize>] {
        &self.site_ion
    }

    pub fn displacements_ticks(&self) -> &[[i64; 3]] {
        &self.displacements_ticks
    }

    pub fn hop_counts(&self) -> &[u64] {
        &self.hop_counts
    }

    pub fn total_hops(&self) -> u128 {
        self.hop_counts.iter().copied().map(u128::from).sum()
    }

    pub fn ion_displacement_m(&self, lattice: &PeriodicLattice, ion: usize) -> Option<[f64; 3]> {
        self.displacements_ticks
            .get(ion)
            .map(|&d| lattice.displacement_m(d))
    }

    /// Mean single-ion tracer squared displacement `mean_i ||dr_i||^2` in `m^2`.
    pub fn mean_tracer_msd_m2(&self, lattice: &PeriodicLattice) -> f64 {
        let n = self.sites.len();
        if n == 0 {
            return 0.0;
        }
        let sum_sq: f64 = self
            .displacements_ticks
            .iter()
            .map(|&d| lattice.squared_displacement_m2(d))
            .sum();
        sum_sq / (n as f64)
    }

    /// Normalized collective charge/center-of-mass squared displacement `||sum_i dr_i||^2 / N` in `m^2`.
    pub fn collective_msd_m2(&self, lattice: &PeriodicLattice) -> f64 {
        let n = self.sites.len();
        if n == 0 {
            return 0.0;
        }
        let mut total_m = [0.0_f64; 3];
        for &d in &self.displacements_ticks {
            let dr = lattice.displacement_m(d);
            total_m[0] += dr[0];
            total_m[1] += dr[1];
            total_m[2] += dr[2];
        }
        vec3_norm_sq(total_m) / (n as f64)
    }
}

#[derive(Clone, Debug, PartialEq)]
enum RateTable {
    Uniform(f64),
    Table {
        type_stride: usize,
        pathway_count: usize,
        rates: Vec<f64>,
    },
}

/// Multi-ion site-exclusion lattice-hopping KMC model over [`PeriodicLattice`].
#[derive(Clone, Debug, PartialEq)]
pub struct LatticeHopping {
    lattice: PeriodicLattice,
    rates: RateTable,
}

impl LatticeHopping {
    pub fn homogeneous(
        lattice: PeriodicLattice,
        jump_rate: f64,
    ) -> Result<Self, LatticeHoppingError> {
        if !jump_rate.is_finite() || jump_rate < 0.0 {
            return Err(LatticeHoppingError::InvalidJumpRate(jump_rate));
        }
        let rate = if jump_rate == 0.0 { 0.0 } else { jump_rate };
        Ok(Self {
            lattice,
            rates: RateTable::Uniform(rate),
        })
    }

    /// Precomputes transition rates from site-type energies (`eV`), pathway parameters, and `temperature_k` (`K`)
    /// using symmetric KRA barriers `max(E_mig + (E_to - E_from) / 2, 0, E_to - E_from)`.
    pub fn with_energetics(
        lattice: PeriodicLattice,
        site_energies_ev: Vec<f64>,
        pathways: Vec<PathwayParams>,
        temperature_k: f64,
    ) -> Result<Self, LatticeHoppingError> {
        if !temperature_k.is_finite() || temperature_k <= 0.0 {
            return Err(LatticeHoppingError::InvalidTemperature(temperature_k));
        }
        for (site_type, &energy_ev) in site_energies_ev.iter().enumerate() {
            if !energy_ev.is_finite() {
                return Err(LatticeHoppingError::InvalidSiteEnergy {
                    site_type,
                    energy_ev,
                });
            }
        }
        for (pathway, p) in pathways.iter().enumerate() {
            if !p.attempt_frequency_hz.is_finite() || p.attempt_frequency_hz <= 0.0 {
                return Err(LatticeHoppingError::InvalidAttemptFrequency {
                    pathway,
                    frequency_hz: p.attempt_frequency_hz,
                });
            }
            if !p.migration_barrier_ev.is_finite() || p.migration_barrier_ev < 0.0 {
                return Err(LatticeHoppingError::InvalidMigrationBarrier {
                    pathway,
                    barrier_ev: p.migration_barrier_ev,
                });
            }
        }
        for (site, &st) in lattice.site_types.iter().enumerate() {
            if st >= site_energies_ev.len() {
                return Err(LatticeHoppingError::SiteTypeOutOfRange {
                    site,
                    site_type: st,
                    type_count: site_energies_ev.len(),
                });
            }
        }
        for hops in &lattice.adjacency {
            for hop in hops {
                if hop.pathway >= pathways.len() {
                    return Err(LatticeHoppingError::PathwayOutOfRange {
                        pathway: hop.pathway,
                        pathway_count: pathways.len(),
                    });
                }
            }
        }

        let type_stride = site_energies_ev.len();
        let pathway_count = pathways.len();
        let table_len = pathway_count
            .checked_mul(type_stride)
            .and_then(|n| n.checked_mul(type_stride))
            .ok_or(LatticeHoppingError::InvalidMetric)?;
        let mut rates = Vec::new();
        if rates.try_reserve_exact(table_len).is_err() {
            return Err(LatticeHoppingError::InvalidMetric);
        }

        let kbt = BOLTZMANN_EV_PER_K * temperature_k;
        for p in &pathways {
            for &e_from in &site_energies_ev {
                for &e_to in &site_energies_ev {
                    let delta_e = e_to - e_from;
                    let raw = p.migration_barrier_ev + 0.5 * delta_e;
                    let barrier = raw.max(0.0).max(delta_e);
                    let rate = if barrier == 0.0 {
                        p.attempt_frequency_hz
                    } else {
                        let exponent = -barrier / kbt;
                        let factor = exponent.exp();
                        if factor > 0.0 {
                            p.attempt_frequency_hz * factor
                        } else {
                            (p.attempt_frequency_hz.ln() + exponent).exp()
                        }
                    };
                    if !rate.is_finite() || rate < 0.0 {
                        return Err(LatticeHoppingError::InvalidJumpRate(rate));
                    }
                    rates.push(if rate == 0.0 { 0.0 } else { rate });
                }
            }
        }

        Ok(Self {
            lattice,
            rates: RateTable::Table {
                type_stride,
                pathway_count,
                rates,
            },
        })
    }

    pub fn lattice(&self) -> &PeriodicLattice {
        &self.lattice
    }

    pub fn transition_rate(&self, from_type: usize, to_type: usize, pathway: usize) -> Option<f64> {
        match &self.rates {
            RateTable::Uniform(rate) => {
                if from_type < self.lattice.type_count
                    && to_type < self.lattice.type_count
                    && pathway < self.lattice.pathway_count
                {
                    Some(*rate)
                } else {
                    None
                }
            }
            RateTable::Table {
                type_stride,
                pathway_count,
                rates,
            } => {
                if from_type >= *type_stride || to_type >= *type_stride || pathway >= *pathway_count
                {
                    return None;
                }
                let idx = (pathway * type_stride + from_type) * type_stride + to_type;
                rates.get(idx).copied()
            }
        }
    }

    #[inline]
    fn transition_rate_for_hop(&self, from_type: usize, to_type: usize, pathway: usize) -> f64 {
        match &self.rates {
            RateTable::Uniform(rate) => *rate,
            RateTable::Table {
                type_stride, rates, ..
            } => rates[(pathway * type_stride + from_type) * type_stride + to_type],
        }
    }

    pub fn initial_state(
        &self,
        ion_sites: &[usize],
    ) -> Result<LatticeHoppingState, LatticeHoppingError> {
        let site_count = self.lattice.site_count();
        if ion_sites.is_empty() {
            return Err(LatticeHoppingError::ZeroIons);
        }
        if ion_sites.len() > site_count {
            return Err(LatticeHoppingError::TooManyIons {
                ions: ion_sites.len(),
                sites: site_count,
            });
        }
        let slots = ion_sites
            .len()
            .checked_mul(self.lattice.max_degree)
            .ok_or(LatticeHoppingError::TooManySlots(usize::MAX))?;
        SumTree::check_len(slots).map_err(|_| LatticeHoppingError::TooManySlots(slots))?;

        let mut site_ion = vec![None; site_count];
        for (ion, &site) in ion_sites.iter().enumerate() {
            if site >= site_count {
                return Err(LatticeHoppingError::SiteOutOfRange {
                    site,
                    sites: site_count,
                });
            }
            if site_ion[site].is_some() {
                return Err(LatticeHoppingError::DuplicateIonSite(site));
            }
            site_ion[site] = Some(ion);
        }
        Ok(LatticeHoppingState {
            site_ion,
            sites: ion_sites.to_vec(),
            origins: ion_sites.to_vec(),
            displacements_ticks: vec![[0, 0, 0]; ion_sites.len()],
            hop_counts: vec![0; ion_sites.len()],
            last_moved_from: None,
        })
    }

    pub fn random_state_with_rng(
        &self,
        ion_count: usize,
        rng: &mut KmcRng,
    ) -> Result<LatticeHoppingState, LatticeHoppingError> {
        let site_count = self.lattice.site_count();
        if ion_count == 0 {
            return Err(LatticeHoppingError::ZeroIons);
        }
        if ion_count > site_count {
            return Err(LatticeHoppingError::TooManyIons {
                ions: ion_count,
                sites: site_count,
            });
        }
        let mut all: Vec<usize> = (0..site_count).collect();
        for i in 0..ion_count {
            let j = i + rng.index(site_count - i);
            all.swap(i, j);
        }
        all.truncate(ion_count);
        self.initial_state(&all)
    }

    pub fn check_invariants(&self, state: &LatticeHoppingState) -> Result<(), LatticeHoppingError> {
        let site_count = self.lattice.site_count();
        if state.site_ion.len() != site_count {
            return Err(LatticeHoppingError::StateMismatch(
                "site_ion length disagrees with lattice site_count",
            ));
        }
        let n = state.sites.len();
        if n == 0 {
            return Err(LatticeHoppingError::ZeroIons);
        }
        if state.origins.len() != n
            || state.displacements_ticks.len() != n
            || state.hop_counts.len() != n
        {
            return Err(LatticeHoppingError::StateMismatch(
                "per-ion state vectors have unequal lengths",
            ));
        }
        let mut seen_origins = vec![false; site_count];
        for ion in 0..n {
            for site in [state.sites[ion], state.origins[ion]] {
                if site >= site_count {
                    return Err(LatticeHoppingError::SiteOutOfRange {
                        site,
                        sites: site_count,
                    });
                }
            }
            if state.site_ion[state.sites[ion]] != Some(ion) {
                return Err(LatticeHoppingError::StateMismatch(
                    "site_ion occupancy map disagrees with sites vector",
                ));
            }
            if seen_origins[state.origins[ion]] {
                return Err(LatticeHoppingError::DuplicateIonSite(state.origins[ion]));
            }
            seen_origins[state.origins[ion]] = true;

            let q0 = self.lattice.site_coords_ticks[state.origins[ion]];
            let q = self.lattice.site_coords_ticks[state.sites[ion]];
            let d = state.displacements_ticks[ion];
            if state.hop_counts[ion] == 0 && d != [0, 0, 0] {
                return Err(LatticeHoppingError::StateMismatch(
                    "ion with zero hops has non-zero displacement",
                ));
            }
            for a in 0..3 {
                let sum = i128::from(q0[a]) + i128::from(d[a]);
                let expected = if self.lattice.periodic_axes[a] {
                    sum.rem_euclid(i128::from(self.lattice.period_ticks[a])) as i64
                } else {
                    if sum != i128::from(q[a]) {
                        return Err(LatticeHoppingError::StateMismatch(
                            "non-periodic unwrapped displacement disagrees with current coordinate",
                        ));
                    }
                    q[a]
                };
                if expected != q[a] {
                    return Err(LatticeHoppingError::StateMismatch(
                        "periodic wrapped displacement disagrees with current coordinate",
                    ));
                }
            }
        }
        let occupied = state.site_ion.iter().filter(|x| x.is_some()).count();
        if occupied != n {
            return Err(LatticeHoppingError::StateMismatch(
                "site_ion contains unexpected occupied entries",
            ));
        }
        if let Some((moved_ion, prev_site)) = state.last_moved_from
            && (moved_ion >= n || prev_site >= site_count)
        {
            return Err(LatticeHoppingError::StateMismatch(
                "last_moved_from indices out of range",
            ));
        }
        Ok(())
    }

    #[inline]
    fn hop_rate(&self, state: &LatticeHoppingState, site: usize, branch: usize) -> f64 {
        let Some(hop) = self.lattice.adjacency[site].get(branch) else {
            return 0.0;
        };
        if state.site_ion[hop.target].is_some() {
            return 0.0;
        }
        let from_type = self.lattice.site_types[site];
        let to_type = self.lattice.site_types[hop.target];
        self.transition_rate_for_hop(from_type, to_type, hop.pathway)
    }
}

impl Model for LatticeHopping {
    type State = LatticeHoppingState;
    type Event = IonHop;
    type Error = LatticeHoppingError;

    fn enumerate_events(&self, state: &Self::State, emit: &mut dyn FnMut(Self::Event, f64)) {
        let site_count = self.lattice.site_count();
        if state.site_ion.len() != site_count
            || state.displacements_ticks.len() != state.sites.len()
            || state.hop_counts.len() != state.sites.len()
            || state.sites.iter().any(|&s| s >= site_count)
        {
            emit(IonHop { ion: 0, branch: 0 }, f64::NAN);
            return;
        }
        for (ion, &site) in state.sites.iter().enumerate() {
            let deg = self.lattice.degree(site);
            for branch in 0..deg {
                let rate = self.hop_rate(state, site, branch);
                if rate != 0.0 {
                    emit(IonHop { ion, branch }, rate);
                }
            }
        }
    }

    fn apply(&self, state: &mut Self::State, event: &Self::Event) -> Result<(), Self::Error> {
        let _ = self.apply_reversible(state, event)?;
        Ok(())
    }
}

impl ReversibleModel for LatticeHopping {
    type Undo = IonHopUndo;

    fn apply_reversible(
        &self,
        state: &mut Self::State,
        event: &Self::Event,
    ) -> Result<Self::Undo, Self::Error> {
        let ion = event.ion;
        let ions = state.sites.len();
        if ion >= ions {
            return Err(LatticeHoppingError::IonOutOfRange { ion, ions });
        }
        if state.site_ion.len() != self.lattice.site_count()
            || state.displacements_ticks.len() != ions
            || state.hop_counts.len() != ions
        {
            return Err(LatticeHoppingError::StateMismatch(
                "state dimensions disagree with lattice or ion count",
            ));
        }
        let from_site = state.sites[ion];
        if from_site >= self.lattice.site_count() || state.site_ion[from_site] != Some(ion) {
            return Err(LatticeHoppingError::StateMismatch(
                "ion site disagrees with site_ion occupancy map",
            ));
        }
        let hops = &self.lattice.adjacency[from_site];
        let hop = hops
            .get(event.branch)
            .ok_or(LatticeHoppingError::BranchOutOfRange {
                ion,
                branch: event.branch,
                degree: hops.len(),
            })?;
        if state.site_ion[hop.target].is_some() {
            return Err(LatticeHoppingError::TargetOccupied {
                ion,
                target: hop.target,
            });
        }
        let prev_disp = state.displacements_ticks[ion];
        let next_disp = [
            prev_disp[0]
                .checked_add(hop.delta_ticks[0])
                .ok_or(LatticeHoppingError::DisplacementOverflow { ion })?,
            prev_disp[1]
                .checked_add(hop.delta_ticks[1])
                .ok_or(LatticeHoppingError::DisplacementOverflow { ion })?,
            prev_disp[2]
                .checked_add(hop.delta_ticks[2])
                .ok_or(LatticeHoppingError::DisplacementOverflow { ion })?,
        ];
        let next_hops = state.hop_counts[ion]
            .checked_add(1)
            .ok_or(LatticeHoppingError::HopCountOverflow { ion })?;

        let prev_last_moved_from = state.last_moved_from;
        state.site_ion[from_site] = None;
        state.site_ion[hop.target] = Some(ion);
        state.sites[ion] = hop.target;
        state.displacements_ticks[ion] = next_disp;
        state.hop_counts[ion] = next_hops;
        state.last_moved_from = Some((ion, from_site));

        Ok(IonHopUndo {
            ion,
            from_site,
            to_site: hop.target,
            prev_disp,
            prev_last_moved_from,
        })
    }

    fn undo(&self, state: &mut Self::State, undo: Self::Undo) {
        if undo.ion >= state.sites.len()
            || undo.from_site >= state.site_ion.len()
            || undo.to_site >= state.site_ion.len()
        {
            return;
        }
        state.site_ion[undo.to_site] = None;
        state.site_ion[undo.from_site] = Some(undo.ion);
        state.sites[undo.ion] = undo.from_site;
        state.displacements_ticks[undo.ion] = undo.prev_disp;
        state.hop_counts[undo.ion] -= 1;
        state.last_moved_from = undo.prev_last_moved_from;
    }
}

impl IncrementalModel for LatticeHopping {
    fn slot_count(&self, state: &Self::State) -> usize {
        state.sites.len() * self.lattice.max_degree
    }

    fn slot_rate(&self, state: &Self::State, slot: usize) -> f64 {
        let stride = self.lattice.max_degree;
        if stride == 0 {
            return f64::NAN;
        }
        let ion = slot / stride;
        let branch = slot % stride;
        let Some(&site) = state.sites.get(ion) else {
            return f64::NAN;
        };
        if site >= self.lattice.site_count() || state.site_ion.len() != self.lattice.site_count() {
            return f64::NAN;
        }
        self.hop_rate(state, site, branch)
    }

    fn slot_event(&self, _state: &Self::State, slot: usize) -> Self::Event {
        let stride = self.lattice.max_degree;
        if stride == 0 {
            return IonHop {
                ion: usize::MAX,
                branch: usize::MAX,
            };
        }
        IonHop {
            ion: slot / stride,
            branch: slot % stride,
        }
    }

    fn affected_slots(&self, state: &Self::State, event: &Self::Event, out: &mut Vec<usize>) {
        let stride = self.lattice.max_degree;
        if stride == 0 {
            return;
        }
        let Some((moved_ion, old_site)) = state.last_moved_from else {
            out.extend(0..self.slot_count(state));
            return;
        };
        if moved_ion != event.ion || event.ion >= state.sites.len() {
            out.extend(0..self.slot_count(state));
            return;
        }
        let new_site = state.sites[event.ion];
        let base = event.ion * stride;
        out.extend((0..stride).map(|b| base + b));
        for site in [old_site, new_site] {
            for hop in self.lattice.hops(site) {
                if let Some(other) = state.site_ion[hop.target]
                    && other != event.ion
                {
                    out.push(other * stride + hop.reverse_branch);
                }
            }
        }
        out.sort_unstable();
        out.dedup();
    }
}

impl Checkpointable for LatticeHopping {
    const MODEL_ID: &'static str = "okrates.lattice_hopping.v1";

    fn encode_params(&self, out: &mut Vec<u8>) {
        for axis in 0..3 {
            out.extend_from_slice(&(self.lattice.cells[axis] as u64).to_le_bytes());
            out.extend_from_slice(&(self.lattice.sub_ticks[axis] as u64).to_le_bytes());
            out.push(u8::from(self.lattice.periodic_axes[axis]));
            for component in self.lattice.tick_vectors_m[axis] {
                out.extend_from_slice(&component.to_bits().to_le_bytes());
            }
        }
        out.extend_from_slice(&(self.lattice.site_count() as u64).to_le_bytes());
        for site in 0..self.lattice.site_count() {
            for q in self.lattice.site_coords_ticks[site] {
                out.extend_from_slice(&q.to_le_bytes());
            }
            out.extend_from_slice(&(self.lattice.site_types[site] as u64).to_le_bytes());
            out.extend_from_slice(&(self.lattice.adjacency[site].len() as u64).to_le_bytes());
            for hop in &self.lattice.adjacency[site] {
                out.extend_from_slice(&(hop.target as u64).to_le_bytes());
                for d in hop.delta_ticks {
                    out.extend_from_slice(&d.to_le_bytes());
                }
                out.extend_from_slice(&(hop.pathway as u64).to_le_bytes());
                out.extend_from_slice(&(hop.reverse_branch as u64).to_le_bytes());
            }
        }
        match &self.rates {
            RateTable::Uniform(rate) => {
                out.push(0);
                out.extend_from_slice(&rate.to_bits().to_le_bytes());
            }
            RateTable::Table {
                type_stride,
                pathway_count,
                rates,
            } => {
                out.push(1);
                out.extend_from_slice(&(*type_stride as u64).to_le_bytes());
                out.extend_from_slice(&(*pathway_count as u64).to_le_bytes());
                for &r in rates {
                    out.extend_from_slice(&r.to_bits().to_le_bytes());
                }
            }
        }
    }

    fn encode_state(&self, state: &Self::State, out: &mut Vec<u8>) {
        out.extend_from_slice(&(state.sites.len() as u64).to_le_bytes());
        let (moved_ion, moved_site) = match state.last_moved_from {
            Some((ion, site)) => (ion as u64, site as u64),
            None => (u64::MAX, u64::MAX),
        };
        out.extend_from_slice(&moved_ion.to_le_bytes());
        out.extend_from_slice(&moved_site.to_le_bytes());
        for ion in 0..state.sites.len() {
            out.extend_from_slice(&(state.sites[ion] as u64).to_le_bytes());
            out.extend_from_slice(&(state.origins[ion] as u64).to_le_bytes());
            for d in state.displacements_ticks[ion] {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&state.hop_counts[ion].to_le_bytes());
        }
    }

    fn decode_state(&self, bytes: &[u8]) -> Result<Self::State, Box<dyn Error + Send + Sync>> {
        let (chunks, remainder) = bytes.as_chunks::<8>();
        if !remainder.is_empty() {
            return Err(format!(
                "lattice hopping state length {} is not a multiple of 8",
                bytes.len()
            )
            .into());
        }
        let mut words = chunks.iter().copied();
        let raw_n = words
            .next()
            .map(u64::from_le_bytes)
            .ok_or("missing ion count word")?;
        let n = usize::try_from(raw_n)
            .ok()
            .filter(|&n| n >= 1 && n <= self.lattice.site_count())
            .ok_or_else(|| format!("invalid ion count {raw_n}"))?;
        let expected_words = n
            .checked_mul(6)
            .and_then(|w| w.checked_add(3))
            .ok_or("lattice hopping state size overflow")?;
        if chunks.len() != expected_words {
            return Err(format!(
                "lattice hopping state must be {} bytes, got {}",
                expected_words * 8,
                bytes.len()
            )
            .into());
        }

        let moved_ion = words
            .next()
            .map(u64::from_le_bytes)
            .ok_or("missing moved_ion word")?;
        let moved_site = words
            .next()
            .map(u64::from_le_bytes)
            .ok_or("missing moved_site word")?;
        let site_count = self.lattice.site_count();
        let last_moved_from = match (moved_ion, moved_site) {
            (u64::MAX, u64::MAX) => None,
            (i, s) => {
                let ion = usize::try_from(i)
                    .ok()
                    .filter(|&idx| idx < n)
                    .ok_or("invalid last_moved_from ion index")?;
                let site = usize::try_from(s)
                    .ok()
                    .filter(|&idx| idx < site_count)
                    .ok_or("invalid last_moved_from site index")?;
                Some((ion, site))
            }
        };

        let parse_site = |raw: [u8; 8]| -> Result<usize, LatticeHoppingError> {
            let val = u64::from_le_bytes(raw);
            usize::try_from(val).ok().filter(|&s| s < site_count).ok_or(
                LatticeHoppingError::SiteOutOfRange {
                    site: usize::try_from(val).unwrap_or(usize::MAX),
                    sites: site_count,
                },
            )
        };

        let mut sites = Vec::with_capacity(n);
        let mut origins = Vec::with_capacity(n);
        let mut displacements_ticks = Vec::with_capacity(n);
        let mut hop_counts = Vec::with_capacity(n);
        for _ in 0..n {
            sites.push(parse_site(words.next().ok_or("truncated site word")?)?);
            origins.push(parse_site(words.next().ok_or("truncated origin word")?)?);
            let dx = i64::from_le_bytes(words.next().ok_or("truncated dx word")?);
            let dy = i64::from_le_bytes(words.next().ok_or("truncated dy word")?);
            let dz = i64::from_le_bytes(words.next().ok_or("truncated dz word")?);
            displacements_ticks.push([dx, dy, dz]);
            hop_counts.push(u64::from_le_bytes(
                words.next().ok_or("truncated hop count word")?,
            ));
        }

        let mut state = self.initial_state(&sites)?;
        state.origins = origins;
        state.displacements_ticks = displacements_ticks;
        state.hop_counts = hop_counts;
        state.last_moved_from = last_moved_from;
        self.check_invariants(&state)?;
        Ok(state)
    }
}
