use super::*;
use search_tree::*;
use state::State;
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};

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
    let result = ptr.compare_exchange(
        std::ptr::null_mut(),
        v as *const _ as *mut _,
        Ordering::Relaxed,
        Ordering::Relaxed,
    );
    match result {
        Ok(_) => None,
        Err(p) => unsafe { Some(&*p) },
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

impl ApproxTable
where
    State: TranspositionHash,
{
    pub fn insert<'a>(&'a self, key: &State, value: &'a SearchNode) -> Option<&'a SearchNode> {
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
            let key_here = entry.k.load(Ordering::Relaxed);
            if key_here == my_hash {
                let value_here = entry.v.load(Ordering::Relaxed);
                if !value_here.is_null() {
                    return unsafe { Some(&*value_here) };
                }
                return get_or_write(&entry.v, value);
            }
            if key_here == 0 {
                let key_here =
                    entry
                        .k
                        .compare_exchange(0, my_hash, Ordering::Relaxed, Ordering::Relaxed);
                self.size.fetch_add(1, Ordering::Relaxed);
                return match key_here {
                    Ok(_) => get_or_write(&entry.v, value),
                    Err(v) if v == my_hash => get_or_write(&entry.v, value),
                    _ => None,
                };
            }
            posn += inc;
            posn &= self.mask;
        }
        None
    }
    pub fn lookup<'a>(&'a self, key: &State) -> Option<&'a SearchNode> {
        let my_hash = key.hash();
        let mut posn = my_hash as usize & self.mask;
        for inc in 1..(PROBE_LIMIT + 1) {
            let entry = unsafe { self.arr.get_unchecked(posn) };
            let key_here = entry.k.load(Ordering::Relaxed);
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
