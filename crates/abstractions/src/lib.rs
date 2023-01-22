use std::fmt::{Display, Formatter};
use std::ops::{Deref, Mul, Range};

#[derive(Default, Debug, Copy, Clone)]
pub struct NumVectors(usize);

#[derive(Default, Debug, Copy, Clone)]
pub struct NumDimensions(usize);

impl NumVectors {
    #[inline(always)]
    pub const fn range(&self) -> Range<usize> {
        0..self.0
    }

    #[inline(always)]
    pub const fn into_inner(self) -> usize {
        self.0
    }
}

impl NumDimensions {
    #[inline(always)]
    pub const fn range(&self) -> Range<usize> {
        0..self.0
    }

    #[inline(always)]
    pub const fn into_inner(self) -> usize {
        self.0
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

impl IntoIterator for NumDimensions {
    type Item = usize;
    type IntoIter = Range<usize>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.range()
    }
}

impl From<usize> for NumDimensions {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl From<usize> for NumVectors {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl From<u32> for NumDimensions {
    fn from(value: u32) -> Self {
        Self(value as _)
    }
}

impl From<u32> for NumVectors {
    fn from(value: u32) -> Self {
        Self(value as _)
    }
}

impl From<i32> for NumDimensions {
    fn from(value: i32) -> Self {
        Self(value as _)
    }
}

impl From<i32> for NumVectors {
    fn from(value: i32) -> Self {
        Self(value as _)
    }
}

impl Into<usize> for NumDimensions {
    fn into(self) -> usize {
        self.0
    }
}

impl Into<usize> for NumVectors {
    fn into(self) -> usize {
        self.0
    }
}

impl AsRef<usize> for NumDimensions {
    fn as_ref(&self) -> &usize {
        &self.0
    }
}

impl AsRef<usize> for NumVectors {
    fn as_ref(&self) -> &usize {
        &self.0
    }
}

impl Deref for NumDimensions {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for NumVectors {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Mul<NumDimensions> for NumVectors {
    type Output = usize;

    fn mul(self, rhs: NumDimensions) -> Self::Output {
        self.0 * rhs.0
    }
}

impl Mul<NumVectors> for NumDimensions {
    type Output = usize;

    fn mul(self, rhs: NumVectors) -> Self::Output {
        self.0 * rhs.0
    }
}

impl Mul<usize> for NumVectors {
    type Output = usize;

    fn mul(self, rhs: usize) -> Self::Output {
        self.0 * rhs
    }
}

impl Mul<usize> for NumDimensions {
    type Output = usize;

    fn mul(self, rhs: usize) -> Self::Output {
        self.0 * rhs
    }
}

impl Mul<NumVectors> for usize {
    type Output = usize;

    fn mul(self, rhs: NumVectors) -> Self::Output {
        self * rhs.0
    }
}

impl Mul<NumDimensions> for usize {
    type Output = usize;

    fn mul(self, rhs: NumDimensions) -> Self::Output {
        self * rhs.0
    }
}

impl Display for NumVectors {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Display for NumDimensions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
