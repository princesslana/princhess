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

pub struct Arena {
    owned: Mutex<Vec<MmapMut>>,
    max_chunks: usize,
    allocators: HashSet<u64>,
}

impl Arena {
    pub fn new(max_size_mb: usize) -> Self {
        let max_chunks = (max_size_mb << 20) / CHUNK_SIZE;
        Self {
            owned: Mutex::default(),
            max_chunks,
            allocators: HashSet::default(),
        }
    }

    fn owned_len(&self) -> usize {
        self.owned.lock().unwrap().len()
    }

    pub fn full(&self) -> usize {
        self.owned_len() * 1000 / self.max_chunks
    }

    pub fn is_full(&self) -> bool {
        self.owned_len() > self.max_chunks
    }

    fn give_mmap(&self, mut map: MmapMut) -> Result<&mut [u8], Error> {
        if self.owned_len() > self.max_chunks {
            return Err(Error::Full);
        }

        let result = ptr::addr_of_mut!(*map);
        self.owned.lock().unwrap().push(map);
        unsafe { Ok(&mut *result) }
    }

    fn alloc_chunk(&self, id: u64) -> Result<&mut [u8], Error> {
        let mmap = self.give_mmap(MmapMut::map_anon(CHUNK_SIZE).unwrap());
        let _ = self.allocators.insert(id);
        mmap
    }

    // This is a combination of give_mmap and alloc_chunk that always succeeds.
    // This means not checking if we exceed the max chunks.
    // This is because we always want creating of an allocator to succeed, even if already full.
    #[allow(clippy::mut_from_ref)]
    fn allocator_chunk(&self, id: u64) -> &mut [u8] {
        let mut mmap = MmapMut::map_anon(CHUNK_SIZE).unwrap();

        let result = ptr::addr_of_mut!(*mmap);

        self.owned.lock().unwrap().push(mmap);

        let _ = self.allocators.insert(id);

        unsafe { &mut *result }
    }

    pub fn allocator(&self) -> Allocator {
        let id = IDS.fetch_add(1, Ordering::Relaxed);

        Allocator {
            id,
            arena: self,
            memory: UnsafeCell::new(self.allocator_chunk(id)),
        }
    }

    pub fn is_allocator_valid(&self, id: u64) -> bool {
        self.allocators.contains(&id)
    }

    pub fn clear(&self) {
        self.allocators.clear();
        self.owned.lock().unwrap().clear();
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
