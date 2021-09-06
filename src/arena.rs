extern crate memmap;
extern crate pod;

use memmap::MmapMut;
use pod::Pod;
use std::cell::UnsafeCell;
use std::collections::LinkedList;
use std::mem;
use std::ops::DerefMut;
use std::sync::Mutex;

#[derive(Debug)]
pub enum ArenaError {
    Full,
}

const CHUNK_SIZE: usize = 1 << 21; // 2MB

pub struct Arena {
    owned_mappings: Mutex<LinkedList<MmapMut>>,
    max_chunks: usize,
}

impl Arena {
    pub fn new(max_size_mb: usize) -> Self {
        let max_chunks = max_size_mb << 20 / CHUNK_SIZE;
        Self {
            owned_mappings: Default::default(),
            max_chunks,
        }
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
    fn alloc_chunk(&self) -> Result<&mut [u8], ArenaError> {
        self.give_mmap(MmapMut::map_anon(CHUNK_SIZE).unwrap())
    }
    pub fn allocator(&self) -> ArenaAllocator {
        ArenaAllocator {
            arena: self,
            memory: UnsafeCell::new(self.alloc_chunk().expect("Could not create allocator")),
        }
    }
}

pub struct ArenaAllocator<'a> {
    arena: &'a Arena,
    memory: UnsafeCell<&'a mut [u8]>,
}

const ALIGN: usize = 8;

impl<'a> ArenaAllocator<'a> {
    fn get_memory(&self, sz: usize) -> Result<&'a mut [u8], ArenaError> {
        let memory = unsafe { &mut *self.memory.get() };
        if sz <= memory.len() {
            let (left, right) = memory.split_at_mut(sz);
            unsafe { *self.memory.get() = right };
            Ok(left)
        } else if sz > CHUNK_SIZE {
            Err(ArenaError::Full)
        } else {
            *memory = self.arena.alloc_chunk()?;
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
