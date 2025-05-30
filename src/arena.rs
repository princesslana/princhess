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
    generation: AtomicU64,
}

impl Arena {
    pub fn new(max_size_mb: usize) -> Self {
        let max = max_size_mb / CHUNK_SIZE_MB;

        Self {
            chunks: Mutex::new(Chunks::with_capacity(max)),
            max,
            generation: AtomicU64::new(1),
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
    fn alloc_chunk(&self) -> Result<&mut Chunk, Error> {
        let mut chunks = self.chunks.lock().unwrap();

        let idx = chunks.used;

        if idx >= self.max {
            return Err(Error::Full);
        }

        chunks.used += 1;

        let result = ptr::addr_of_mut!(chunks.chunks[idx]);
        Ok(unsafe { &mut *result })
    }

    pub fn allocator(&self) -> Allocator {
        let current_generation = self.generation.load(Ordering::Acquire);
        Allocator {
            generation: AtomicU64::new(current_generation),
            arena: self,
            memory: UnsafeCell::new(self.alloc_chunk().unwrap()),
        }
    }

    pub fn invalid_allocator(&self) -> Allocator {
        let dummy = DUMMY_CHUNK.as_ptr() as *mut Chunk;

        Allocator {
            generation: AtomicU64::new(0),
            arena: self,
            memory: UnsafeCell::new(unsafe { &mut *dummy }),
        }
    }

    pub fn clear(&self) {
        self.chunks.lock().unwrap().used = 0;
        self.generation.fetch_add(1, Ordering::Release);
    }
}

pub struct Allocator<'a> {
    generation: AtomicU64,
    arena: &'a Arena,
    memory: UnsafeCell<&'a mut [u8]>,
}

const ALIGN: usize = 8;

impl<'a> Allocator<'a> {
    fn get_memory(&self, sz: usize) -> Result<&'a mut [u8], Error> {
        let memory = unsafe { &mut *self.memory.get() };

        let current_arena_generation = self.arena.generation.load(Ordering::Acquire);
        let allocator_generation = self.generation.load(Ordering::Acquire);

        if allocator_generation != current_arena_generation {
            // Arena was cleared, this allocator is stale. Get a new chunk and update generation.
            unsafe { *self.memory.get() = self.arena.alloc_chunk()? };
            self.generation
                .store(current_arena_generation, Ordering::Release);
            self.get_memory(sz)
        } else if sz <= memory.len() {
            let (left, right) = memory.split_at_mut(sz);
            unsafe { *self.memory.get() = right };
            Ok(left)
        } else if sz > CHUNK_SIZE {
            Err(Error::Full)
        } else {
            // Current chunk is full, get a new one.
            unsafe { *self.memory.get() = self.arena.alloc_chunk()? };
            self.get_memory(sz)
        }
    }

    pub fn alloc_contiguous_pair<T, U>(
        &self,
        num_t_items: usize,
    ) -> Result<(&'a mut [T], &'a mut U), Error>
    where
        T: Sized,
        U: Sized,
    {
        let size_of_t = mem::size_of::<T>();
        let align_of_t = mem::align_of::<T>();
        let size_of_u = mem::size_of::<U>();
        let align_of_u = mem::align_of::<U>();

        assert!(
            ALIGN % align_of_t == 0,
            "Type T alignment not compatible with ALIGN"
        );
        assert!(
            ALIGN % align_of_u == 0,
            "Type U alignment not compatible with ALIGN"
        );

        let t_total_byte_size = num_t_items * size_of_t;

        let u_offset = (t_total_byte_size + align_of_u - 1) & !(align_of_u - 1);

        let total_allocation_size = u_offset + size_of_u;

        let raw_memory = self.get_memory(total_allocation_size)?;

        let t_ptr = raw_memory.as_mut_ptr().cast::<T>();
        let t_slice_ref = unsafe { slice::from_raw_parts_mut(t_ptr, num_t_items) };

        let u_ptr = unsafe { raw_memory.as_mut_ptr().add(u_offset).cast::<U>() };
        let u_ref = unsafe { &mut *u_ptr };

        Ok((t_slice_ref, u_ref))
    }
}
