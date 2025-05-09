use scc::HashSet;
use std::alloc::{self, Layout};
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

const BYTES_PER_MB: usize = 1 << 20;
const CHUNK_SIZE_MB: usize = 2;
const CHUNK_SIZE: usize = BYTES_PER_MB * CHUNK_SIZE_MB;

type Chunk = [u8; CHUNK_SIZE];

static DUMMY_CHUNK: Chunk = [0; CHUNK_SIZE];

static IDS: AtomicU64 = AtomicU64::new(0);

struct Chunks {
    pub chunks: Box<[Chunk]>,
    pub used: usize,
}

impl Chunks {
    fn with_capacity(capacity: usize) -> Self {
        let layout = Layout::array::<Chunk>(capacity).unwrap();
        let chunks = unsafe {
            let raw = alloc::alloc(layout);
            if raw.is_null() {
                alloc::handle_alloc_error(layout);
            }

            let slice = slice::from_raw_parts_mut(raw.cast(), capacity);

            Box::from_raw(slice)
        };

        Self { chunks, used: 0 }
    }
}
pub struct Arena {
    chunks: Mutex<Chunks>,
    max: usize,
    allocators: HashSet<u64>,
}

impl Arena {
    pub fn new(max_size_mb: usize) -> Self {
        let max = max_size_mb / CHUNK_SIZE_MB;

        Self {
            chunks: Mutex::new(Chunks::with_capacity(max)),
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
    fn alloc_chunk(&self, id: u64) -> Result<&mut Chunk, Error> {
        let mut chunks = self.chunks.lock().unwrap();

        let idx = chunks.used;

        if idx >= self.max {
            return Err(Error::Full);
        }

        chunks.used += 1;

        let _ = self.allocators.insert(id);

        let result = ptr::addr_of_mut!(chunks.chunks[idx]);
        Ok(unsafe { &mut *result })
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

        let dummy = DUMMY_CHUNK.as_ptr() as *mut Chunk;

        Allocator {
            id,
            arena: self,
            memory: UnsafeCell::new(unsafe { &mut *dummy }),
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
        let x = x + ((!x + 1) % ALIGN);
        let x = self.get_memory(x)?;
        Ok(unsafe { &mut *(x.as_mut_ptr().cast::<T>()) })
    }

    pub fn alloc_slice<T>(&self, sz: usize) -> Result<&'a mut [T], Error> {
        assert!(ALIGN % mem::align_of::<T>() == 0);
        let x = mem::size_of::<T>();
        let x = x + ((!x + 1) % ALIGN);
        let x = self.get_memory(x * sz)?;
        Ok(unsafe { slice::from_raw_parts_mut(x.as_mut_ptr().cast::<T>(), sz) })
    }
}
