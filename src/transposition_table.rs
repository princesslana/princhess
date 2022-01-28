use super::*;
use search_tree::*;
use state::State;
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};

pub unsafe trait TranspositionTable: Sync + Sized {
    /// **If this function inserts a value, it must return `None`.** Failure to follow
    /// this rule will lead to memory safety violation.
    ///
    /// Attempts to insert a key/value pair.
    ///
    /// If the key is not present, the table *may* insert it. If the table does
    /// not insert it, the table may either return `None` or a reference to another
    /// value existing in the table. (The latter is allowed so that the table doesn't
    /// necessarily need to handle hash collisions, but it will negatively affect the accuracy
    /// of the search.)
    ///
    /// If the key is present, the table may either:
    /// - Leave the table unchanged and return `Some(reference to associated value)`.
    /// - Leave the table unchanged and return `None`.
    ///
    /// The table *may* choose to replace old values.
    /// The table is *not* responsible for dropping values that are replaced.
    fn insert<'a>(&'a self, key: &State, value: &'a SearchNode) -> Option<&'a SearchNode>;

    /// Looks up a key.
    ///
    /// If the key is not present, the table *should almost always* return `None`.
    ///
    /// If the key is present, the table *may return either* `None` or a reference
    /// to the associated value.
    fn lookup<'a>(&'a self, key: &State) -> Option<&'a SearchNode>;
}

unsafe impl TranspositionTable for () {
    fn insert<'a>(&'a self, _: &State, _: &'a SearchNode) -> Option<&'a SearchNode> {
        None
    }

    fn lookup<'a>(&'a self, _: &State) -> Option<&'a SearchNode> {
        None
    }
}

pub trait TranspositionHash {
    fn hash(&self) -> u64;
}

pub struct ApproxQuadraticProbingHashTable<K: TranspositionHash, V> {
    arr: Box<[Entry16<K, V>]>,
    capacity: usize,
    mask: usize,
    size: AtomicUsize,
}

struct Entry16<K: TranspositionHash, V> {
    k: AtomicU64,
    v: AtomicPtr<V>,
    _marker: std::marker::PhantomData<K>,
}

impl<K: TranspositionHash, V> Default for Entry16<K, V> {
    fn default() -> Self {
        Self {
            k: Default::default(),
            v: Default::default(),
            _marker: Default::default(),
        }
    }
}
impl<K: TranspositionHash, V> Clone for Entry16<K, V> {
    fn clone(&self) -> Self {
        Self {
            k: AtomicU64::new(self.k.load(Ordering::Relaxed)),
            v: AtomicPtr::new(self.v.load(Ordering::Relaxed)),
            _marker: Default::default(),
        }
    }
}

impl<K: TranspositionHash, V> ApproxQuadraticProbingHashTable<K, V> {
    pub fn new(capacity: usize) -> Self {
        assert!(std::mem::size_of::<Entry16<K, V>>() <= 16);
        assert!(
            capacity.count_ones() == 1,
            "the capacity must be a power of 2"
        );
        let arr = vec![Entry16::default(); capacity].into_boxed_slice();
        let mask = capacity - 1;
        Self {
            arr,
            mask,
            capacity,
            size: AtomicUsize::default(),
        }
    }
    pub fn enough_to_hold(num: usize) -> Self {
        let mut capacity = 1;
        while capacity * 2 < num * 3 {
            capacity <<= 1;
        }
        Self::new(capacity)
    }
}

unsafe impl<K: TranspositionHash, V> Sync for ApproxQuadraticProbingHashTable<K, V> {}
unsafe impl<K: TranspositionHash, V> Send for ApproxQuadraticProbingHashTable<K, V> {}

pub type ApproxTable = ApproxQuadraticProbingHashTable<State, SearchNode>;

fn get_or_write<'a, V>(ptr: &AtomicPtr<V>, v: &'a V) -> Option<&'a V> {
    match ptr.compare_exchange_weak(
        std::ptr::null_mut(),
        v as *const _ as *mut _,
        Ordering::Relaxed,
        Ordering::Relaxed,
    ) {
        Ok(_) => Some(&*v),
        Err(_) => None,
    }
}

fn convert<'a, V>(ptr: *const V) -> Option<&'a V> {
    if ptr.is_null() {
        None
    } else {
        unsafe { Some(&*ptr) }
    }
}

const PROBE_LIMIT: usize = 16;

unsafe impl TranspositionTable for ApproxTable
where
    State: TranspositionHash,
{
    fn insert<'a>(&'a self, key: &State, value: &'a SearchNode) -> Option<&'a SearchNode> {
        if self.size.load(Ordering::Relaxed) * 3 > self.capacity * 2 {
            return self.lookup(key);
        }
        let my_hash = key.hash();
        if my_hash == 0 {
            return None;
        }
        let mut posn = my_hash as usize & self.mask;
        for inc in 1..(PROBE_LIMIT + 1) {
            let entry = unsafe { self.arr.get_unchecked(posn) };
            let key_here = entry.k.load(Ordering::Relaxed) as u64;
            if key_here == my_hash {
                let value_here = entry.v.load(Ordering::Relaxed);
                if !value_here.is_null() {
                    return unsafe { Some(&*value_here) };
                }
                return get_or_write(&entry.v, value);
            }
            if key_here == 0 {
                self.size.fetch_add(1, Ordering::Relaxed);
                match entry.k.compare_exchange_weak(
                    0,
                    my_hash as u64,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return get_or_write(&entry.v, value),
                    Err(h) if h == my_hash => return get_or_write(&entry.v, value),
                    Err(_) => (),
                }
            }
            posn += inc;
            posn &= self.mask;
        }
        None
    }
    fn lookup<'a>(&'a self, key: &State) -> Option<&'a SearchNode> {
        let my_hash = key.hash();
        let mut posn = my_hash as usize & self.mask;
        for inc in 1..(PROBE_LIMIT + 1) {
            let entry = unsafe { self.arr.get_unchecked(posn) };
            let key_here = entry.k.load(Ordering::Relaxed) as u64;
            if key_here == my_hash {
                return convert(entry.v.load(Ordering::Relaxed));
            }
            if key_here == 0 {
                return None;
            }
            posn += inc;
            posn &= self.mask;
        }
        None
    }
}
