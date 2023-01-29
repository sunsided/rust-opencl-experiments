//! Provide type-safe specification of vector counts.

#![deny(missing_docs)]
#![deny(clippy::missing_docs_in_private_items)]

use std::cmp::Ordering;
use std::fmt::{Display, Formatter};
use std::num::{NonZeroU32, NonZeroUsize};
use std::ops::{Deref, Mul, Range};

/// Describes the number of vectors.
#[derive(Default, Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct NumVectors(
    /// The wrapped number of vectors.
    usize,
);

impl NumVectors {
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

impl Display for NumVectors {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl IntoIterator for NumVectors {
    type Item = usize;
    type IntoIter = Range<usize>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.range()
    }
}

impl From<usize> for NumVectors {
    #[inline(always)]
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl From<NonZeroUsize> for NumVectors {
    #[inline(always)]
    fn from(value: NonZeroUsize) -> Self {
        Self(value.get())
    }
}

impl From<u32> for NumVectors {
    #[inline(always)]
    fn from(value: u32) -> Self {
        Self(value as _)
    }
}

impl From<NonZeroU32> for NumVectors {
    #[inline(always)]
    fn from(value: NonZeroU32) -> Self {
        Self(value.get() as _)
    }
}

impl From<NumVectors> for usize {
    #[inline(always)]
    fn from(value: NumVectors) -> Self {
        value.0
    }
}

impl AsRef<usize> for NumVectors {
    #[inline(always)]
    fn as_ref(&self) -> &usize {
        &self.0
    }
}

impl Deref for NumVectors {
    type Target = usize;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Mul<usize> for NumVectors {
    type Output = usize;
    #[inline(always)]
    fn mul(self, rhs: usize) -> Self::Output {
        self.0 * rhs
    }
}

impl Mul<NumVectors> for usize {
    type Output = usize;

    #[inline(always)]
    fn mul(self, rhs: NumVectors) -> Self::Output {
        self * rhs.0
    }
}

impl PartialEq<usize> for NumVectors {
    #[inline(always)]
    fn eq(&self, other: &usize) -> bool {
        self.0.eq(other)
    }
}

impl PartialEq<NumVectors> for usize {
    #[inline(always)]
    fn eq(&self, other: &NumVectors) -> bool {
        other.eq(self)
    }
}

impl PartialOrd<usize> for NumVectors {
    #[inline(always)]
    fn partial_cmp(&self, other: &usize) -> Option<Ordering> {
        self.0.partial_cmp(other)
    }
}

impl PartialOrd<NumVectors> for usize {
    #[inline(always)]
    fn partial_cmp(&self, other: &NumVectors) -> Option<Ordering> {
        self.partial_cmp(&other.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_works() {
        assert_eq!(NumVectors::from(10u32) * 42usize, NumVectors::from(420u32));

        // This multiplication is commutative.
        assert_eq!(10usize * NumVectors::from(42u32), NumVectors::from(420u32));
    }

    #[test]
    fn partial_eq_usize_works() {
        assert!(0 < NumVectors::from(42u32));
        assert!(NumVectors::from(42u32) > 0);
        assert!(43 > NumVectors::from(42u32));
        assert!(NumVectors::from(42u32) < 43);
    }
}
