//! Random vector generation for testing.

use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro128Plus;

/// An `f32` vector generator based on the Xoshiro-128+ RNG
/// to deterministically generate pseudo-random vectors based on a seed.
#[derive(Debug)]
pub struct Vecgen {
    rng: Xoshiro128Plus,
}

unsafe impl Send for Vecgen {}

impl Vecgen {
    /// Generates a new [`Vecgen`] instance seeded by system entropy.
    pub fn new_from_entropy() -> Self {
        Self {
            rng: Xoshiro128Plus::from_entropy(),
        }
    }

    /// Generates a new [`Vecgen`] instance seeded by the specified value.
    ///
    /// ## Arguments
    /// * `seed` - The seed value to use.
    pub fn new_from_seed(seed: u64) -> Self {
        Self {
            rng: Xoshiro128Plus::seed_from_u64(seed),
        }
    }

    /// Fills a slice with random floating point values.
    #[inline(always)]
    pub fn fill<Q: AsMut<[f32]> + ?Sized>(&mut self, dest: &mut Q) {
        self.rng.fill(dest.as_mut())
    }

    /// Consumes self and fills a buffer with random floating point values,
    /// returning the buffer.
    #[inline(always)]
    pub fn into_filled<Q: AsMut<[f32]>>(mut self, mut dest: Q) -> Q {
        self.rng.fill(dest.as_mut());
        dest
    }

    /// Forks this rng to create a new instance capable of creating
    /// 2^64 non-overlapping floating-point numbers.
    pub fn fork(&self) -> Self {
        let mut rng = self.rng.clone();
        rng.jump();
        Self { rng }
    }
}

impl Default for Vecgen {
    fn default() -> Self {
        Self::new_from_entropy()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn trivial_works() {
        let seed = 1337;
        let mut vector = [0f32; 4096];
        Xoshiro128Plus::seed_from_u64(seed).fill(&mut vector);
        assert_relative_eq!(vector[0], 0.87221956f32, epsilon = 1e-5);
    }

    #[test]
    fn fill_works() {
        let mut rng = Vecgen::new_from_seed(1337);
        let mut vector = [0f32; 4096];
        rng.fill(&mut vector);
        assert_relative_eq!(vector[0], 0.87221956f32, epsilon = 1e-5);
    }

    #[test]
    fn fork_works() {
        let rng = Vecgen::new_from_seed(1337);

        // rng 2 and 3 are forked from the same source, so should produce identical results.
        let mut rng2 = rng.fork();
        let mut vector2 = [0f32; 4096];
        rng2.fill(&mut vector2);

        let mut rng3 = rng.fork();
        let mut vector3 = [0f32; 4096];
        rng3.fill(&mut vector3);

        // rng 4 is forked from rng2, so should produce different results from all others.
        let mut rng4 = rng2.fork();
        let mut vector4 = [0f32; 4096];
        rng4.fill(&mut vector4);

        assert_relative_eq!(vector2[0], 0.918626, epsilon = 1e-5);
        assert_relative_eq!(vector3[0], 0.918626, epsilon = 1e-5);
        assert_relative_eq!(vector4[0], 0.40357572, epsilon = 1e-5);
    }
}