extern crate pod;
extern crate memmap;

use pod::Pod;
use std::ops::DerefMut;
use std::sync::Mutex;
use std::collections::LinkedList;
use std::mem;
use std::cell::UnsafeCell;
use memmap::MmapMut;

pub struct Arena {
    owned_slices: Mutex<LinkedList<Box<[u8]>>>,
    owned_mappings: Mutex<LinkedList<MmapMut>>,
}

impl Arena {
    pub fn new() -> Self {
        Self {
            owned_slices: Default::default(),
            owned_mappings: Default::default(),
        }
    }
    fn give_boxed(&self, mut memory: Box<[u8]>) -> &mut [u8] {
        let result = (&mut *memory) as *mut _;
        let mut owned_slices = self.owned_slices.lock().unwrap();
        owned_slices.push_back(memory);
        unsafe {&mut *result}
    }
    fn give_mmap(&self, mut map: MmapMut) -> &mut [u8] {
        let result = map.deref_mut() as *mut _;
        let mut owned_mappings = self.owned_mappings.lock().unwrap();
        owned_mappings.push_back(map);
        unsafe {&mut *result}
    }
    fn alloc(&self, sz: usize) -> &mut [u8] {
        if sz == 1 << 21 {
            self.give_mmap(MmapMut::map_anon(sz).unwrap())
        } else {
            self.give_boxed(vec![0; sz].into_boxed_slice())
        }
    }
    pub fn allocator(&self) -> ArenaAllocator {
        self.allocator_with_chunk_size(1 << 21) // 2 MB
    }
    pub fn allocator_with_chunk_size(&self, chunk_size: usize) -> ArenaAllocator {
        ArenaAllocator {
            arena: self,
            memory: UnsafeCell::new(self.alloc(chunk_size)),
            chunk_size,
        }
    }
}

pub struct ArenaAllocator<'a> {
    arena: &'a Arena,
    chunk_size: usize,
    memory: UnsafeCell<&'a mut [u8]>,
}

const ALIGN: usize = 8;

impl<'a> ArenaAllocator<'a> {
    fn get_memory(&self, sz: usize) -> &'a mut [u8] {
        let memory = unsafe { &mut *self.memory.get() };
        if sz <= memory.len() {
            let (left, right) = memory.split_at_mut(sz);
            unsafe { *self.memory.get() = right };
            left
        } else if sz > self.chunk_size {
            self.arena.alloc(sz)
        } else {
            *memory = self.arena.alloc(self.chunk_size);
            self.get_memory(sz)
        }
    }
    pub fn alloc_one<T: Pod>(&self) -> &'a mut T {
        assert!(ALIGN % mem::align_of::<T>() == 0);
        let x = mem::size_of::<T>();
        let x = x + ((!x + 1) % ALIGN); // TODO fix panic when x=0
        let x = self.get_memory(x);
        let x = T::ref_from_slice_mut(x);
        x.unwrap()
    }
    pub fn alloc_slice<T: Pod>(&self, sz: usize) -> &'a mut [T] {
        assert!(ALIGN % mem::align_of::<T>() == 0);
        let x = mem::size_of::<T>();
        let x = x + ((!x + 1) % ALIGN); // TODO fix panic when x=0
        let x = self.get_memory(x * sz);
        let x = u8::map_slice_mut(x);
        x.unwrap()
    }
}
