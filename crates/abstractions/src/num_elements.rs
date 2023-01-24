use crate::{NumDimensions, NumVectors};
use std::fmt::{Display, Formatter};
use std::num::{NonZeroU32, NonZeroUsize};
use std::ops::{Deref, Mul, Range};

/// Describes a number of elements.
#[derive(Default, Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct NumElements(usize);

impl NumElements {
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

impl IntoIterator for NumElements {
    type Item = usize;
    type IntoIter = Range<usize>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.range()
    }
}

impl From<usize> for NumElements {
    #[inline(always)]
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl From<NonZeroUsize> for NumElements {
    #[inline(always)]
    fn from(value: NonZeroUsize) -> Self {
        Self(value.get())
    }
}

impl From<u32> for NumElements {
    #[inline(always)]
    fn from(value: u32) -> Self {
        Self(value as _)
    }
}

impl From<NonZeroU32> for NumElements {
    #[inline(always)]
    fn from(value: NonZeroU32) -> Self {
        Self(value.get() as _)
    }
}

impl Into<usize> for NumElements {
    #[inline(always)]
    fn into(self) -> usize {
        self.0
    }
}

impl AsRef<usize> for NumElements {
    #[inline(always)]
    fn as_ref(&self) -> &usize {
        &self.0
    }
}

impl Deref for NumElements {
    type Target = usize;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Mul<usize> for NumElements {
    type Output = usize;
    #[inline(always)]
    fn mul(self, rhs: usize) -> Self::Output {
        self.0 * rhs
    }
}

impl Mul<NumElements> for usize {
    type Output = usize;

    #[inline(always)]
    fn mul(self, rhs: NumElements) -> Self::Output {
        self * rhs.0
    }
}

impl Mul<NumVectors> for NumDimensions {
    type Output = NumElements;

    #[inline(always)]
    fn mul(self, rhs: NumVectors) -> Self::Output {
        NumElements(self.get() * rhs.get())
    }
}

impl Mul<NumDimensions> for NumVectors {
    type Output = NumElements;

    #[inline(always)]
    fn mul(self, rhs: NumDimensions) -> Self::Output {
        NumElements(self.get() * rhs.get())
    }
}

impl Display for NumElements {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_works() {
        assert_eq!(
            NumDimensions::from(10u32) * NumVectors::from(42u32),
            NumElements::from(420u32)
        );

        // This multiplication is commutative.
        assert_eq!(
            NumVectors::from(10u32) * NumDimensions::from(42u32),
            NumElements::from(420u32)
        )
    }
}
