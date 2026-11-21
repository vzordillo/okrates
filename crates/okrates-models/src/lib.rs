//! Periodic BCC lattices and kinetic Monte Carlo models for vacancy and alloy diffusion.

#![forbid(unsafe_code)]

pub mod bcc;
pub mod fecr;
pub mod fecr_anneal;
pub mod fecr_diffusion;
pub mod fecr_martinez;
pub mod fecr_ris;
pub mod fecr_ris_segregation;
pub mod fecr_ris_transport;
pub mod lattice_gas;
pub mod lattice_hopping;
pub mod multi_vacancy;
pub mod reaction_network;
pub mod tabulated;
pub mod vacancy;
