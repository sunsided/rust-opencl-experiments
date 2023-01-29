//! Provides memory handling functionality such as aligned buffer allocation
//! and indexed access into vectors.
//!
//! ## Features
//! - `power-of-two-chunks`: Enforces that the memory chunk's size is a power of two.
//!     This may result in improved performance on some hardware but could waste memory
//!     depending on the vector size.
pub mod chunks;
mod errors;
mod memory_view;
mod utils;

pub use errors::InsertVectorError;
