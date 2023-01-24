use std::fmt::{Display, Formatter};
use std::num::NonZeroUsize;

/// A node-local vector ID.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct LocalId(usize);

impl LocalId {
    /// Consumes this instance and returns the wrapped [`usize`].
    ///
    /// ## Example
    /// ```
    /// # use abstractions::LocalId;
    /// assert_eq!(LocalId::from(42).get(), 42)
    /// ```
    #[inline(always)]
    pub const fn get(self) -> usize {
        self.0
    }
}

impl From<usize> for LocalId {
    /// Constructs a [`LocalId`] instance from a [`usize`] value.
    ///
    /// ## Example
    /// ```
    /// # use abstractions::LocalId;
    /// assert_eq!(LocalId::from(42).get(), 42)
    /// ```
    #[inline(always)]
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl From<NonZeroUsize> for LocalId {
    /// Constructs a [`LocalId`] instance from a [`NonZeroUsize`] value.
    ///
    /// ## Example
    /// ```
    /// # use abstractions::LocalId;
    /// assert_eq!(LocalId::from(42).get(), 42)
    /// ```
    #[inline(always)]
    fn from(value: NonZeroUsize) -> Self {
        Self(value.get())
    }
}

impl Into<usize> for LocalId {
    #[inline(always)]
    fn into(self) -> usize {
        self.0
    }
}

impl Display for LocalId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "LocalId({})", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_works() {
        assert_eq!(
            format!("{:?}", LocalId::from(42)),
            String::from("LocalId(42)")
        )
    }

    #[test]
    fn display_works() {
        assert_eq!(
            format!("{:?}", LocalId::from(42)),
            String::from("LocalId(42)")
        )
    }

    #[test]
    fn into_inner_works() {
        assert_eq!(LocalId::from(42).get(), 42)
    }
}
