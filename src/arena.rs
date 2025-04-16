use memmap::MmapMut;
use scc::HashSet;
use std::cell::UnsafeCell;
use std::mem;
use std::ptr;
use std::slice;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

#[derive(Debug)]
pub enum Error {
    Full,
}

const CHUNK_SIZE: usize = 1 << 21; // 2MB

static IDS: AtomicU64 = AtomicU64::new(0);

struct Chunks {
    pub chunks: Vec<MmapMut>,
    pub used: usize,
}

impl Default for Chunks {
    fn default() -> Self {
        Self {
            chunks: Vec::with_capacity(256),
            used: 0,
        }
    }
}

pub struct Arena {
    chunks: Mutex<Chunks>,
    max: usize,
    allocators: HashSet<u64>,
}

impl Arena {
    pub fn new(max_size_mb: usize) -> Self {
        let max = (max_size_mb << 20) / CHUNK_SIZE;

        Self {
            chunks: Mutex::default(),
            max,
            allocators: HashSet::default(),
        }
    }

    fn used(&self) -> usize {
        self.chunks.lock().unwrap().used
    }

    pub fn full(&self) -> usize {
        self.used() * 1000 / self.max
    }

    pub fn is_full(&self) -> bool {
        self.used() >= self.max
    }

    pub fn capacity_remaining(&self) -> usize {
        self.max - self.used()
    }

    #[allow(clippy::mut_from_ref)]
    fn alloc_chunk(&self, id: u64) -> Result<&mut [u8], Error> {
        let mut chunks = self.chunks.lock().unwrap();

        let idx = chunks.used;

        if idx >= self.max {
            return Err(Error::Full);
        }

        chunks.used += 1;

        if chunks.chunks.len() <= idx {
            chunks.chunks.push(MmapMut::map_anon(CHUNK_SIZE).unwrap());
        }

        let _ = self.allocators.insert(id);

        let result = ptr::addr_of_mut!(*chunks.chunks[idx]);
        Ok(unsafe { &mut *result })
    }

    #[allow(clippy::mut_from_ref)]
    fn last_chunk(&self, id: u64) -> &mut [u8] {
        if self.used() == 0 {
            self.alloc_chunk(id).unwrap()
        } else {
            let mut chunks = self.chunks.lock().unwrap();
            let idx = chunks.used - 1;
            unsafe { &mut *ptr::addr_of_mut!(*chunks.chunks[idx]) }
        }
    }

    pub fn allocator(&self) -> Allocator {
        let id = IDS.fetch_add(1, Ordering::Relaxed);

        Allocator {
            id,
            arena: self,
            memory: UnsafeCell::new(self.alloc_chunk(id).unwrap()),
        }
    }

    pub fn invalid_allocator(&self) -> Allocator {
        let id = IDS.fetch_add(1, Ordering::Relaxed);

        Allocator {
            id,
            arena: self,
            memory: UnsafeCell::new(self.last_chunk(id)),
        }
    }

    pub fn is_allocator_valid(&self, id: u64) -> bool {
        self.allocators.contains(&id)
    }

    pub fn clear(&self) {
        self.allocators.clear();
        self.chunks.lock().unwrap().used = 0;
    }
}

pub struct Allocator<'a> {
    id: u64,
    arena: &'a Arena,
    memory: UnsafeCell<&'a mut [u8]>,
}

const ALIGN: usize = 8;

impl<'a> Allocator<'a> {
    fn get_memory(&self, sz: usize) -> Result<&'a mut [u8], Error> {
        let memory = unsafe { &mut *self.memory.get() };

        if !self.arena.is_allocator_valid(self.id) {
            *memory = self.arena.alloc_chunk(self.id)?;
            self.get_memory(sz)
        } else if sz <= memory.len() {
            let (left, right) = memory.split_at_mut(sz);
            unsafe { *self.memory.get() = right };
            Ok(left)
        } else if sz > CHUNK_SIZE {
            Err(Error::Full)
        } else {
            *memory = self.arena.alloc_chunk(self.id)?;
            self.get_memory(sz)
        }
    }

    pub fn alloc_one<T>(&self) -> Result<&'a mut T, Error> {
        assert!(ALIGN % mem::align_of::<T>() == 0);
        let x = mem::size_of::<T>();
        let x = x + ((!x + 1) % ALIGN); // TODO fix panic when x=0
        let x = self.get_memory(x)?;
        Ok(unsafe { &mut *(x.as_mut_ptr().cast::<T>()) })
    }

    pub fn alloc_slice<T>(&self, sz: usize) -> Result<&'a mut [T], Error> {
        assert!(ALIGN % mem::align_of::<T>() == 0);
        let x = mem::size_of::<T>();
        let x = x + ((!x + 1) % ALIGN); // TODO fix panic when x=0
        let x = self.get_memory(x * sz)?;
        Ok(unsafe { slice::from_raw_parts_mut(x.as_mut_ptr().cast::<T>(), sz) })
    }
}
