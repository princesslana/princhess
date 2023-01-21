extern crate memmap;
extern crate pod;

use arc_swap::ArcSwap;
use log::debug;
use memmap::MmapMut;
use pod::Pod;
use std::cell::UnsafeCell;
use std::collections::{HashSet, LinkedList};
use std::mem;
use std::ops::DerefMut;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

#[derive(Debug)]
pub enum ArenaError {
    Full,
}

const CHUNK_SIZE: usize = 1 << 21; // 2MB

static IDS: AtomicU64 = AtomicU64::new(0);

pub struct Arena {
    owned_mappings: Mutex<LinkedList<MmapMut>>,
    max_chunks: usize,
    allocators: ArcSwap<HashSet<u64>>,
}

impl Arena {
    pub fn new(max_size_mb: usize) -> Self {
        let max_chunks = (max_size_mb << 20) / CHUNK_SIZE;
        debug!(
            "Creating Arena of {}mb = {} chunks",
            max_size_mb, max_chunks
        );
        Self {
            owned_mappings: Default::default(),
            max_chunks,
            allocators: Default::default(),
        }
    }

    pub fn full(&self) -> bool {
        let owned_mappings = self.owned_mappings.lock().unwrap();
        owned_mappings.len() > self.max_chunks
    }

    fn give_mmap(&self, mut map: MmapMut) -> Result<&mut [u8], ArenaError> {
        let result = map.deref_mut() as *mut _;
        let mut owned_mappings = self.owned_mappings.lock().unwrap();
        if owned_mappings.len() > self.max_chunks {
            Err(ArenaError::Full)
        } else {
            owned_mappings.push_back(map);
            unsafe { Ok(&mut *result) }
        }
    }

    fn alloc_chunk(&self, id: u64) -> Result<&mut [u8], ArenaError> {
        let allocators = self.allocators.load();
        if !allocators.contains(&id) {
            self.allocators.rcu(|als| {
                let mut als = HashSet::clone(als);
                als.insert(id);
                als
            });
        }

        self.give_mmap(MmapMut::map_anon(CHUNK_SIZE).unwrap())
    }

    // This is a combination of give_mmap and alloc_chunk that always succeeds.
    // This means not checking if we exceed the max chunks.
    // This is because we always want creating of an allocator to succeed, even if already full.
    #[allow(clippy::mut_from_ref)]
    fn allocator_chunk(&self, id: u64) -> &mut [u8] {
        let allocators = self.allocators.load();
        if !allocators.contains(&id) {
            self.allocators.rcu(|als| {
                let mut als = HashSet::clone(als);
                als.insert(id);
                als
            });
        }

        let mut mmap = MmapMut::map_anon(CHUNK_SIZE).unwrap();

        let result = mmap.deref_mut() as *mut _;
        let mut owned_mappings = self.owned_mappings.lock().unwrap();

        owned_mappings.push_back(mmap);
        unsafe { &mut *result }
    }

    pub fn allocator(&self) -> ArenaAllocator {
        let id = IDS.fetch_add(1, Ordering::SeqCst);

        ArenaAllocator {
            id,
            arena: self,
            memory: UnsafeCell::new(self.allocator_chunk(id)),
        }
    }

    pub fn is_allocator_valid(&self, id: u64) -> bool {
        self.allocators.load().contains(&id)
    }

    pub fn clear(&self) {
        let mut owned_mappings = self.owned_mappings.lock().unwrap();
        owned_mappings.clear();

        self.allocators.store(HashSet::new().into());
    }
}

pub struct ArenaAllocator<'a> {
    id: u64,
    arena: &'a Arena,
    memory: UnsafeCell<&'a mut [u8]>,
}

const ALIGN: usize = 8;

impl<'a> ArenaAllocator<'a> {
    fn get_memory(&self, sz: usize) -> Result<&'a mut [u8], ArenaError> {
        let memory = unsafe { &mut *self.memory.get() };

        if !self.arena.is_allocator_valid(self.id) {
            debug!("Invalid allocator {}", self.id);
            *memory = self.arena.alloc_chunk(self.id)?;
            self.get_memory(sz)
        } else if sz <= memory.len() {
            let (left, right) = memory.split_at_mut(sz);
            unsafe { *self.memory.get() = right };
            Ok(left)
        } else if sz > CHUNK_SIZE {
            debug!("sz > CHUNK_SIZE, {} > {}", sz, CHUNK_SIZE);
            Err(ArenaError::Full)
        } else {
            *memory = self.arena.alloc_chunk(self.id)?;
            self.get_memory(sz)
        }
    }

    pub fn alloc_one<T: Pod>(&self) -> Result<&'a mut T, ArenaError> {
        assert!(ALIGN % mem::align_of::<T>() == 0);
        let x = mem::size_of::<T>();
        let x = x + ((!x + 1) % ALIGN); // TODO fix panic when x=0
        let x = self.get_memory(x)?;
        let x = T::ref_from_slice_mut(x);
        Ok(x.unwrap())
    }

    pub fn alloc_slice<T: Pod>(&self, sz: usize) -> Result<&'a mut [T], ArenaError> {
        assert!(ALIGN % mem::align_of::<T>() == 0);
        let x = mem::size_of::<T>();
        let x = x + ((!x + 1) % ALIGN); // TODO fix panic when x=0
        let x = self.get_memory(x * sz)?;
        let x = u8::map_slice_mut(x);
        Ok(x.unwrap())
    }
}
