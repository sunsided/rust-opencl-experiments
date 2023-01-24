pub trait Alignment {
    /// Determines whether the address is 256-bit aligned.
    fn is_32byte_aligned(&self) -> bool;

    /// Determines whether the address is 512-bit aligned.
    fn is_64byte_aligned(&self) -> bool;
}

impl Alignment for usize {
    fn is_32byte_aligned(&self) -> bool {
        0 == (self & (32usize - 1))
    }

    fn is_64byte_aligned(&self) -> bool {
        0 == (self & (64usize - 1))
    }
}

impl Alignment for *const f32 {
    fn is_32byte_aligned(&self) -> bool {
        ((*self) as usize).is_32byte_aligned()
    }

    fn is_64byte_aligned(&self) -> bool {
        ((*self) as usize).is_64byte_aligned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avx2_usize_works() {
        // AVX-2 has 256 bit wide registers, spanning 8 f32 elements.
        assert!(0usize.is_32byte_aligned());
        assert!((8 * 4usize).is_32byte_aligned());

        assert!(!1usize.is_32byte_aligned());
        assert!(!(7 * 4usize).is_32byte_aligned());
        assert!(!(9 * 4usize).is_32byte_aligned());
    }

    #[test]
    fn avx512_usize_works() {
        // AVX-512 has 512 bit wide registers, spanning 16 f32 elements.
        assert!(0usize.is_64byte_aligned());
        assert!((16 * 4usize).is_64byte_aligned());

        assert!(!1usize.is_64byte_aligned());
        assert!(!(15 * 4usize).is_64byte_aligned());
        assert!(!(17 * 4usize).is_64byte_aligned());
    }

    #[test]
    fn avx2_ptr_works() {
        assert!(!(16 as *const f32).is_32byte_aligned());
        assert!((32 as *const f32).is_32byte_aligned());
        assert!((128 as *const f32).is_32byte_aligned());
        assert!(!(127 as *const f32).is_32byte_aligned());
    }

    #[test]
    fn avx512_ptr_works() {
        assert!(!(16 as *const f32).is_64byte_aligned());
        assert!(!(32 as *const f32).is_64byte_aligned());
        assert!((64 as *const f32).is_64byte_aligned());
        assert!(!(127 as *const f32).is_64byte_aligned());
        assert!((128 as *const f32).is_64byte_aligned());
    }
}
