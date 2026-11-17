//! Periodic body-centred cubic lattice topology.

use std::error::Error;
use std::fmt;

/// Nearest-neighbour displacements in half-lattice units, in lexicographic order.
pub const DISPLACEMENTS: [[i64; 3]; 8] = [
    [-1, -1, -1],
    [-1, -1, 1],
    [-1, 1, -1],
    [-1, 1, 1],
    [1, -1, -1],
    [1, -1, 1],
    [1, 1, -1],
    [1, 1, 1],
];

/// Second-nearest-neighbour displacements in half-lattice units.
pub const SECOND_DISPLACEMENTS: [[i64; 3]; 6] = [
    [-2, 0, 0],
    [2, 0, 0],
    [0, -2, 0],
    [0, 2, 0],
    [0, 0, -2],
    [0, 0, 2],
];

/// A periodic BCC lattice with precomputed first- and second-neighbour tables.
#[derive(Clone, Debug)]
pub struct BccLattice {
    cells: [usize; 3],
    neighbors: Vec<[usize; 8]>,
    second_neighbors: Vec<[usize; 6]>,
}

impl BccLattice {
    /// Creates a lattice with at least two conventional cells along each axis.
    pub fn new(cells: [usize; 3]) -> Result<Self, LatticeError> {
        if cells.iter().any(|&dimension| dimension < 2) {
            return Err(LatticeError::TooSmall { cells });
        }

        let mut lattice = Self {
            cells,
            neighbors: Vec::with_capacity(2 * cells[0] * cells[1] * cells[2]),
            second_neighbors: Vec::with_capacity(2 * cells[0] * cells[1] * cells[2]),
        };
        for site in 0..lattice.site_count() {
            let position = lattice.position(site);
            let neighbors = std::array::from_fn(|direction| {
                let target = std::array::from_fn(|axis| {
                    (position[axis] + DISPLACEMENTS[direction][axis])
                        .rem_euclid(2 * cells[axis] as i64)
                });
                lattice
                    .site_at(target)
                    .expect("BCC neighbour positions share a parity")
            });
            lattice.neighbors.push(neighbors);
            let second = std::array::from_fn(|direction| {
                let target = std::array::from_fn(|axis| {
                    position[axis] + SECOND_DISPLACEMENTS[direction][axis]
                });
                lattice
                    .site_at(target)
                    .expect("second-neighbour positions share a parity")
            });
            lattice.second_neighbors.push(second);
        }
        Ok(lattice)
    }

    /// Returns the conventional cell dimensions.
    pub fn cells(&self) -> [usize; 3] {
        self.cells
    }

    /// Returns the number of lattice sites.
    pub fn site_count(&self) -> usize {
        2 * self.cells[0] * self.cells[1] * self.cells[2]
    }

    /// Returns the neighbour of `site` along `direction`.
    pub fn neighbor(&self, site: usize, direction: usize) -> usize {
        self.neighbors[site][direction]
    }

    /// Returns the eight nearest neighbours of `site` in displacement order.
    pub fn neighbors(&self, site: usize) -> &[usize; 8] {
        &self.neighbors[site]
    }

    /// Returns the six second-nearest neighbours of `site` in [`SECOND_DISPLACEMENTS`] order.
    ///
    /// With two cells along an axis the − and + entries on that axis coincide.
    pub fn second_neighbors(&self, site: usize) -> &[usize; 6] {
        &self.second_neighbors[site]
    }

    /// Returns a site's position in half-lattice units.
    pub fn position(&self, site: usize) -> [i64; 3] {
        let [lx, ly, lz] = self.cells;
        let cells_per_sublattice = lx * ly * lz;
        let sublattice = site / cells_per_sublattice;
        let within_sublattice = site % cells_per_sublattice;
        let i = within_sublattice / (ly * lz);
        let j = (within_sublattice / lz) % ly;
        let k = within_sublattice % lz;
        [
            (2 * i + sublattice) as i64,
            (2 * j + sublattice) as i64,
            (2 * k + sublattice) as i64,
        ]
    }

    /// Wraps a position into the box and returns its site if its components share parity.
    pub fn site_at(&self, position: [i64; 3]) -> Option<usize> {
        let wrapped: [i64; 3] =
            std::array::from_fn(|axis| position[axis].rem_euclid(2 * self.cells[axis] as i64));
        let sublattice = wrapped[0].rem_euclid(2);
        if wrapped[1].rem_euclid(2) != sublattice || wrapped[2].rem_euclid(2) != sublattice {
            return None;
        }

        let i = ((wrapped[0] - sublattice) / 2) as usize;
        let j = ((wrapped[1] - sublattice) / 2) as usize;
        let k = ((wrapped[2] - sublattice) / 2) as usize;
        let [lx, ly, lz] = self.cells;
        Some(((sublattice as usize * lx + i) * ly + j) * lz + k)
    }
}

/// An error constructing a BCC lattice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatticeError {
    /// At least one conventional cell dimension was less than two.
    TooSmall {
        /// The rejected cell dimensions.
        cells: [usize; 3],
    },
}

impl fmt::Display for LatticeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooSmall { cells } => {
                write!(
                    f,
                    "BCC cell dimensions must all be at least 2, got {cells:?}"
                )
            }
        }
    }
}

impl Error for LatticeError {}

#[cfg(test)]
mod tests {
    use super::{BccLattice, DISPLACEMENTS, LatticeError};

    #[test]
    fn rejects_small_cells() {
        for cells in [[1, 2, 2], [2, 1, 2], [2, 2, 1], [2, 2, 0]] {
            assert_eq!(
                BccLattice::new(cells).unwrap_err(),
                LatticeError::TooSmall { cells }
            );
        }
    }

    #[test]
    fn site_count() {
        for (cells, expected) in [([2, 2, 2], 16), ([3, 3, 3], 54), ([2, 3, 4], 48)] {
            assert_eq!(BccLattice::new(cells).unwrap().site_count(), expected);
        }
    }

    #[test]
    fn topology_invariants() {
        for cells in [[2, 2, 2], [2, 3, 4], [5, 5, 5]] {
            let lattice = BccLattice::new(cells).unwrap();
            let mut in_degree = vec![0; lattice.site_count()];

            for site in 0..lattice.site_count() {
                let position = lattice.position(site);
                let neighbors = lattice.neighbors(site);
                for direction in 0..8 {
                    let neighbor = neighbors[direction];
                    assert_ne!(neighbor, site);
                    assert!(!neighbors[..direction].contains(&neighbor));
                    in_degree[neighbor] += 1;

                    let expected_position = std::array::from_fn(|axis| {
                        (position[axis] + DISPLACEMENTS[direction][axis])
                            .rem_euclid(2 * cells[axis] as i64)
                    });
                    assert_eq!(lattice.position(neighbor), expected_position);

                    let opposite = DISPLACEMENTS
                        .iter()
                        .position(|candidate| {
                            std::array::from_fn::<_, 3, _>(|axis| -DISPLACEMENTS[direction][axis])
                                == *candidate
                        })
                        .unwrap();
                    assert_eq!(lattice.neighbor(neighbor, opposite), site);
                }
            }

            assert!(in_degree.iter().all(|&count| count == 8));
        }
    }

    #[test]
    fn explicit_wrap() {
        let lattice = BccLattice::new([2, 3, 4]).unwrap();
        let start = lattice.site_at([0, 0, 0]).unwrap();
        let negative = DISPLACEMENTS
            .iter()
            .position(|&displacement| displacement == [-1, -1, -1])
            .unwrap();
        let positive = DISPLACEMENTS
            .iter()
            .position(|&displacement| displacement == [1, 1, 1])
            .unwrap();
        let wrapped = lattice.site_at([3, 5, 7]).unwrap();

        assert_eq!(lattice.neighbor(start, negative), wrapped);
        assert_eq!(lattice.neighbor(wrapped, positive), start);
    }

    #[test]
    fn position_roundtrip() {
        let lattice = BccLattice::new([2, 3, 4]).unwrap();
        for site in 0..lattice.site_count() {
            assert_eq!(lattice.site_at(lattice.position(site)), Some(site));
        }
        assert_eq!(lattice.site_at([0, 0, 1]), None);
        assert_eq!(lattice.site_at([-2, 6, 9]), lattice.site_at([2, 0, 1]));
    }
}
