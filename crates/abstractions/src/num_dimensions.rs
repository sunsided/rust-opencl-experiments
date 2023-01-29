//! Provide type-safe specification of vector dimensionality.

#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]

use std::cmp::Ordering;
use std::fmt::{Display, Formatter};
use std::num::{NonZeroU32, NonZeroUsize};
use std::ops::{Deref, Mul, Range};

// TODO: Enforce that vector dimensionality is a multiple of 16.

/// Describes the dimensionality of a vector, i.e. the number of its entries.
#[derive(Default, Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct NumDimensions(
    /// The wrapped number of dimensions.
    usize,
);

impl NumDimensions {
    /// A 384-dimensional entity.
    pub const DIMS_384: NumDimensions = Self(384);

    /// Generates a range of elements from `0` to the represented number.
    #[inline(always)]
    pub const fn range(&self) -> Range<usize> {
        0..self.0
    }

    /// Gets the underlying value.
    #[inline(always)]
    pub const fn get(self) -> usize {
        self.0
    }
}

impl Display for NumDimensions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl IntoIterator for NumDimensions {
    type Item = usize;
    type IntoIter = Range<usize>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.range()
    }
}

impl From<usize> for NumDimensions {
    #[inline(always)]
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl From<NonZeroUsize> for NumDimensions {
    #[inline(always)]
    fn from(value: NonZeroUsize) -> Self {
        Self(value.get())
    }
}

impl From<u32> for NumDimensions {
    #[inline(always)]
    fn from(value: u32) -> Self {
        Self(value as _)
    }
}

impl From<NonZeroU32> for NumDimensions {
    #[inline(always)]
    fn from(value: NonZeroU32) -> Self {
        Self(value.get() as _)
    }
}

impl From<NumDimensions> for usize {
    #[inline(always)]
    fn from(value: NumDimensions) -> Self {
        value.0
    }
}

impl AsRef<usize> for NumDimensions {
    #[inline(always)]
    fn as_ref(&self) -> &usize {
        &self.0
    }
}

impl Deref for NumDimensions {
    type Target = usize;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Mul<usize> for NumDimensions {
    type Output = usize;

    #[inline(always)]
    fn mul(self, rhs: usize) -> Self::Output {
        self.0 * rhs
    }
}

impl Mul<NumDimensions> for usize {
    type Output = usize;

    #[inline(always)]
    fn mul(self, rhs: NumDimensions) -> Self::Output {
        self * rhs.0
    }
}

impl PartialEq<usize> for NumDimensions {
    #[inline(always)]
    fn eq(&self, other: &usize) -> bool {
        self.0.eq(other)
    }
}

impl PartialEq<NumDimensions> for usize {
    #[inline(always)]
    fn eq(&self, other: &NumDimensions) -> bool {
        other.eq(self)
    }
}

impl PartialOrd<usize> for NumDimensions {
    #[inline(always)]
    fn partial_cmp(&self, other: &usize) -> Option<Ordering> {
        self.0.partial_cmp(other)
    }
}

impl PartialOrd<NumDimensions> for usize {
    #[inline(always)]
    fn partial_cmp(&self, other: &NumDimensions) -> Option<Ordering> {
        self.partial_cmp(&other.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_works() {
        assert_eq!(
            NumDimensions::from(10u32) * 42usize,
            NumDimensions::from(420u32)
        );

        // This multiplication is commutative.
        assert_eq!(
            10usize * NumDimensions::from(42u32),
            NumDimensions::from(420u32)
        );
    }

    #[test]
    fn partial_eq_usize_works() {
        assert!(0 < NumDimensions::from(42u32));
        assert!(NumDimensions::from(42u32) > 0);
        assert!(43 > NumDimensions::from(42u32));
        assert!(NumDimensions::from(42u32) < 43);
    }
}
