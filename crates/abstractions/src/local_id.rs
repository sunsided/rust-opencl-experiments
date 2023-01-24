use std::fmt::{Debug, Display, Formatter};
use std::num::NonZeroUsize;

/// A node-local vector ID. The wrapped type is a non-zero, allowing
/// the internal use of the value `0` as a marker for unused elements.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct LocalId(NonZeroUsize);

impl LocalId {
    /// Creates an ID without checking whether the provided
    /// value is non-zero.
    ///
    /// ## Example
    /// ```
    /// # use std::num::NonZeroUsize;
    /// # use abstractions::LocalId;
    /// let expected = NonZeroUsize::try_from(42).unwrap();
    /// assert_eq!(unsafe { LocalId::new_unchecked(42) }.get(), expected)
    /// ```
    #[inline(always)]
    pub const unsafe fn new_unchecked(id: usize) -> Self {
        Self(NonZeroUsize::new_unchecked(id))
    }

    /// Consumes this instance and returns the wrapped [`NonZeroUsize`].
    ///
    /// ## Example
    /// ```
    /// # use std::num::NonZeroUsize;
    /// # use abstractions::LocalId;
    /// let expected = NonZeroUsize::try_from(42).unwrap();
    /// assert_eq!(LocalId::try_from(42).unwrap().get(), expected)
    /// ```
    #[inline(always)]
    pub const fn get(self) -> NonZeroUsize {
        self.0
    }

    /// Consumes this instance and returns the wrapped [`usize`].
    ///
    /// ## Example
    /// ```
    /// # use abstractions::LocalId;
    /// assert_eq!(LocalId::try_from(42).unwrap().into_inner(), 42)
    /// ```
    #[inline(always)]
    pub const fn into_inner(self) -> usize {
        self.0.get()
    }
}

impl TryFrom<usize> for LocalId {
    type Error = <NonZeroUsize as TryFrom<usize>>::Error;

    /// Constructs a [`LocalId`] instance from a [`usize`] value.
    ///
    /// ## Example
    /// ```
    /// # use std::num::NonZeroUsize;
    /// # use abstractions::LocalId;
    /// let expected = NonZeroUsize::try_from(42).unwrap();
    /// assert_eq!(LocalId::try_from(42).unwrap().get(), expected)
    /// ```
    #[inline(always)]
    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Ok(Self(NonZeroUsize::try_from(value)?))
    }
}

impl From<NonZeroUsize> for LocalId {
    /// Constructs a [`LocalId`] instance from a [`NonZeroUsize`] value.
    ///
    /// ## Example
    /// ```
    /// # use std::num::NonZeroUsize;
    /// # use abstractions::LocalId;
    /// let expected = NonZeroUsize::try_from(42).unwrap();
    /// assert_eq!(LocalId::from(expected).get(), expected)
    /// ```
    #[inline(always)]
    fn from(value: NonZeroUsize) -> Self {
        Self(value)
    }
}

impl Into<NonZeroUsize> for LocalId {
    #[inline(always)]
    fn into(self) -> NonZeroUsize {
        self.0
    }
}

impl Into<usize> for LocalId {
    #[inline(always)]
    fn into(self) -> usize {
        self.0.get()
    }
}

impl Debug for LocalId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "LocalId({})", self.0.get())
    }
}

impl Display for LocalId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "LocalId({})", self.0.get())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_works() {
        assert_eq!(
            format!("{:?}", LocalId::try_from(42).unwrap()),
            String::from("LocalId(42)")
        )
    }

    #[test]
    fn display_works() {
        assert_eq!(
            format!("{:?}", LocalId::try_from(42).unwrap()),
            String::from("LocalId(42)")
        )
    }

    #[test]
    fn into_inner_works() {
        assert_eq!(LocalId::try_from(42).unwrap().into_inner(), 42)
    }
}
