use abstractions::LocalId;
use std::borrow::Borrow;
use std::collections::BTreeMap;

/// An ID type.
pub trait IdType: Ord + Sized {}

impl IdType for LocalId {}

/// A registry of [`IdType`] to arbitrary values.
///
/// The main use case for this type is keeping track of [`LocalId`] assignments
/// to memory chunks within chunk managers.
pub struct IdRegistry<Key: IdType, Value: Sized> {
    // TODO: Benchmark against https://lib.rs/crates/btree-slab
    map: BTreeMap<Key, Value>,
}

impl<Key, Value> IdRegistry<Key, Value>
where
    Key: IdType,
    Value: Sized,
{
    /// Constructs a new [`IdRegistry`].
    pub const fn new() -> Self {
        Self {
            map: BTreeMap::new(),
        }
    }

    /// Retrieves the element of the specified `key`. Returns [`None`] if no such element exists.
    #[inline(always)]
    pub fn get<Q>(&self, key: &Q) -> Option<&Value>
    where
        Key: Borrow<Q> + Ord,
        Q: Ord,
    {
        self.map.get(key)
    }

    /// Convenience function to test for the existence of the specified `key` in the registry.
    ///
    /// If you need access to the element, prefer the direct use of [`IdRegistry::get`].
    #[inline(always)]
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        Key: Borrow<Q> + Ord,
        Q: Ord,
    {
        self.map.contains_key(key)
    }

    /// Inserts an element into the registry and returns the previous value if it existed.
    #[inline(always)]
    pub fn insert(&mut self, key: Key, value: Value) -> Option<Value> {
        self.map.insert(key, value)
    }

    /// Removes an element from the registry and returns its value.
    #[inline(always)]
    pub fn remove<Q>(&mut self, key: &Q) -> Option<Value>
    where
        Key: Borrow<Q> + Ord,
        Q: Ord,
    {
        self.map.remove(key.borrow())
    }

    /// Gets the number of elements in the registry.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Indicates whether the registry is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl IdType for usize {}

    #[test]
    fn new_map_is_empty() {
        let map: IdRegistry<usize, &str> = IdRegistry::new();
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        assert_eq!(map.get(&0), None);
    }

    #[test]
    fn insert_works() {
        let mut map = IdRegistry::new();
        assert_eq!(map.insert(42, "first"), None);
        assert_eq!(map.insert(42, "second"), Some("first"));
        assert_eq!(map.get(&42), Some(&"second"));
        assert_eq!(map.get(&0), None);
        assert_eq!(map.len(), 1);
        assert!(!map.is_empty());
    }

    #[test]
    fn remove_works() {
        let mut map = IdRegistry::new();
        assert_eq!(map.insert(42, "first"), None);

        assert_eq!(map.remove(&42), Some("first"));
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());

        assert_eq!(map.remove(&42), None);
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
    }
}
