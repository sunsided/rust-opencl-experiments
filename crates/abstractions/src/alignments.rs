#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]

/// Validations for byte boundary alignments of different pointer-like types.
pub trait Alignment {
    /// Determines whether the value represents an address that is aligned to a 256-bit (i.e. 32 byte) boundary.
    fn is_32byte_aligned(&self) -> bool;

    /// Determines whether the value represents an address that is aligned to a 512-bit (i.e. 64 byte) boundary.
    fn is_64byte_aligned(&self) -> bool;
}

/// Determines whether the specified value is a multiple of 32.
#[inline(always)]
const fn is_multiple_of_32(value: usize) -> bool {
    const MASK: usize = 32 - 1;
    0 == value & MASK
}

/// Determines whether the specified value is a multiple of 64.
#[inline(always)]
const fn is_multiple_of_64(value: usize) -> bool {
    const MASK: usize = 64 - 1;
    0 == value & MASK
}

impl<T> Alignment for *const T {
    /// Determines whether the value points to an address that is aligned to a 256-bit (i.e. 32 byte) boundary.
    #[inline(always)]
    fn is_32byte_aligned(&self) -> bool {
        is_multiple_of_32((*self) as usize)
    }

    /// Determines whether the value points to an address that is aligned to a 512-bit (i.e. 64 byte) boundary.
    #[inline(always)]
    fn is_64byte_aligned(&self) -> bool {
        is_multiple_of_64((*self) as usize)
    }
}

impl<T> Alignment for *mut T {
    /// Determines whether the value points to an address that is aligned to a 256-bit (i.e. 32 byte) boundary.
    #[inline(always)]
    fn is_32byte_aligned(&self) -> bool {
        is_multiple_of_32((*self) as usize)
    }

    /// Determines whether the value points to an address that is aligned to a 256-bit (i.e. 32 byte) boundary.
    #[inline(always)]
    fn is_64byte_aligned(&self) -> bool {
        is_multiple_of_64((*self) as usize)
    }
}

impl<T> Alignment for &T {
    /// Determines whether the value refers to an address that is aligned to a 256-bit (i.e. 32 byte) boundary.
    #[inline(always)]
    fn is_32byte_aligned(&self) -> bool {
        ((*self) as *const T).is_32byte_aligned()
    }

    /// Determines whether the value refers to an address that is aligned to a 256-bit (i.e. 32 byte) boundary.
    #[inline(always)]
    fn is_64byte_aligned(&self) -> bool {
        ((*self) as *const T).is_64byte_aligned()
    }
}

impl<T> Alignment for &mut T {
    /// Determines whether the value refers to an address that is aligned to a 256-bit (i.e. 32 byte) boundary.
    #[inline(always)]
    fn is_32byte_aligned(&self) -> bool {
        ((*self) as *const T).is_32byte_aligned()
    }

    /// Determines whether the value refers to an address that is aligned to a 256-bit (i.e. 32 byte) boundary.
    #[inline(always)]
    fn is_64byte_aligned(&self) -> bool {
        ((*self) as *const T).is_64byte_aligned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_multiple_of_32_works() {
        // AVX-2 has 256 bit wide registers, spanning 8 f32 elements.
        assert!(is_multiple_of_32(0usize));
        assert!(is_multiple_of_32(8 * 4usize));

        assert!(!is_multiple_of_32(1usize));
        assert!(!is_multiple_of_32(7 * 4usize));
        assert!(!is_multiple_of_32(9 * 4usize));
    }

    #[test]
    fn is_multiple_of_64_works() {
        // AVX-512 has 512 bit wide registers, spanning 16 f32 elements.
        assert!(is_multiple_of_64(0usize));
        assert!(is_multiple_of_64(16 * 4usize));

        assert!(!is_multiple_of_64(1usize));
        assert!(!is_multiple_of_64(15 * 4usize));
        assert!(!is_multiple_of_64(17 * 4usize));
    }

    #[test]
    fn avx2_ptr_works() {
        test_alignment_32::<f32>();
        test_alignment_32::<f64>();
        test_alignment_32::<core::ffi::c_void>();
    }

    #[test]
    fn avx512_ptr_works() {
        test_alignment_64::<f32>();
        test_alignment_64::<f64>();
        test_alignment_64::<core::ffi::c_void>();
    }

    fn get_addresses_32() -> Vec<(usize, bool)> {
        vec![
            (16, false),
            (32, true),
            (64, true),
            (127, false),
            (128, true),
        ]
    }

    fn get_addresses_64() -> Vec<(usize, bool)> {
        vec![
            (16, false),
            (32, false),
            (64, true),
            (127, false),
            (128, true),
        ]
    }

    /// Implements the test functions `test_alignment_N` for the specified `N`,
    /// assuming that the function `get_addresses_N` exists.
    macro_rules! impl_alignment_tests {
        ($size:literal) => {
            paste::paste! {
                fn [<test_alignment_ $size>]<T>() {
                    for (addr, is_aligned) in [<get_addresses_ $size>] () {
                        const SIZE: usize = $size;
                        let assertion_alignment_msg = if is_aligned { "aligned" } else { "not aligned" };

                        assert_eq!(
                            unsafe { &*(addr as *const T) }.[<is_ $size byte_aligned>](),
                            is_aligned,
                            "Expected immutable reference to address {addr} to be {assertion_alignment_msg} to a {SIZE} byte boundary",
                        );

                        assert_eq!(
                            unsafe { &mut *(addr as *mut T) }.[<is_ $size byte_aligned>](),
                            is_aligned,
                            "Expected mutable reference to address {addr} to be {assertion_alignment_msg} to a {SIZE} byte boundary",
                        );

                        assert_eq!(
                            (addr as *const T).[<is_ $size byte_aligned>](),
                            is_aligned,
                            "Expected constant pointer to address {addr} to be {assertion_alignment_msg} to a {SIZE} byte boundary",
                        );

                        assert_eq!(
                            (addr as *mut T).[<is_ $size byte_aligned>](),
                            is_aligned,
                            "Expected mutable pointer to address {addr} to be {assertion_alignment_msg} to a {SIZE} byte boundary",
                        );
                    }
                }
            }
        };
    }

    impl_alignment_tests!(32);
    impl_alignment_tests!(64);
}
