use std::alloc::{self, Layout};
use std::cell::UnsafeCell;
use std::mem::{self, MaybeUninit};
use std::ops::{Deref, DerefMut};
use std::ptr::{self, NonNull};
use std::slice;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::mem::Align64;

#[derive(Debug)]
pub enum Error {
    Full,
}

const BYTES_PER_MB: usize = 1 << 20;
const CHUNK_SIZE_MB: usize = 2;
const CHUNK_SIZE: usize = BYTES_PER_MB * CHUNK_SIZE_MB;

type Chunk = Align64<[u8; CHUNK_SIZE]>;

// Declared as `static mut` to allow creating a `&mut` reference for `UnsafeCell::new`.
// This is inherently unsafe and should only be done when strictly necessary,
// as it allows mutable global state without synchronization.
// In this context, it's used for an "invalid" allocator that immediately
// re-allocates a valid chunk, so the dummy memory is never actually written to.
static mut DUMMY_CHUNK: Chunk = Align64([0; CHUNK_SIZE]);

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
    fn alloc_chunk(&self) -> Result<&mut [u8], Error> {
        let mut chunks = self.chunks.lock().unwrap();

        let idx = chunks.used;

        if idx >= self.max {
            return Err(Error::Full);
        }

        chunks.used += 1;

        let result = ptr::addr_of_mut!(chunks.chunks[idx]);
        Ok(unsafe { slice::from_raw_parts_mut(result.cast(), CHUNK_SIZE) })
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
        // SAFETY: DUMMY_CHUNK is `static mut`, allowing a mutable reference.
        // This reference is immediately wrapped in `UnsafeCell` and is only
        // used as a placeholder; `get_memory` will re-allocate a valid chunk
        // before any actual writes occur if the generation is mismatched.
        let dummy_ref: &mut [u8] = unsafe { &mut *DUMMY_CHUNK };

        Allocator {
            generation: AtomicU64::new(0), // Set to 0 to ensure re-allocation on first use
            arena: self,
            memory: UnsafeCell::new(dummy_ref),
        }
    }

    pub fn clear(&self) {
        self.chunks.lock().unwrap().used = 0;
        self.generation.fetch_add(1, Ordering::Release);
    }

    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
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

    pub fn alloc_one<T>(&self) -> Result<ArenaRef<MaybeUninit<T>>, Error> {
        assert!(ALIGN % mem::align_of::<T>() == 0);
        let x = mem::size_of::<T>();
        let x = (x + ALIGN - 1) & !(ALIGN - 1);
        let x = self.get_memory(x)?;
        let current_generation = self.arena.generation.load(Ordering::Acquire);
        Ok(ArenaRef {
            ptr: unsafe { NonNull::new_unchecked(x.as_mut_ptr().cast::<MaybeUninit<T>>()) },
            generation: current_generation,
        })
    }

    pub fn alloc_slice<T>(&self, sz: usize) -> Result<ArenaRef<[MaybeUninit<T>]>, Error> {
        assert!(ALIGN % mem::align_of::<T>() == 0);
        let x = mem::size_of::<T>();
        let x = (x + ALIGN - 1) & !(ALIGN - 1);
        let x = self.get_memory(x * sz)?;
        let current_generation = self.arena.generation.load(Ordering::Acquire);
        Ok(ArenaRef {
            ptr: unsafe {
                NonNull::new_unchecked(slice::from_raw_parts_mut(
                    x.as_mut_ptr().cast::<MaybeUninit<T>>(),
                    sz,
                ))
            },
            generation: current_generation,
        })
    }
}

pub struct ArenaRef<T: ?Sized> {
    ptr: NonNull<T>,
    generation: u64,
}

impl<T: ?Sized> ArenaRef<T> {
    #[must_use]
    pub fn from_raw_parts(ptr: NonNull<T>, generation: u64) -> Self {
        Self { ptr, generation }
    }

    #[must_use]
    pub fn into_non_null(self) -> NonNull<T> {
        self.ptr
    }

    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }
}

impl<T> ArenaRef<MaybeUninit<T>> {
    /// Writes a value into the allocated memory and returns an `ArenaRef` to the initialized `T`.
    ///
    /// # Safety
    ///
    /// This function is safe because it takes `self` by value, ensuring that the `MaybeUninit`
    /// wrapper is consumed and replaced by the initialized `T`. The memory is guaranteed to be
    /// properly aligned and sized by the allocator.
    #[must_use]
    pub fn write(self, val: T) -> ArenaRef<T> {
        let ptr = self.ptr.as_ptr();
        unsafe {
            // Prefer writing the raw T for clarity
            ptr::write(ptr.cast::<T>(), val);
            ArenaRef {
                ptr: self.ptr.cast(),
                generation: self.generation,
            }
        }
    }
}

impl<T> ArenaRef<[MaybeUninit<T>]> {
    /// Initializes each element of the slice in place using the provided closure.
    ///
    /// The closure `f` is called for each index, and must return the value to
    /// initialize the element at that index.
    ///
    /// This function is safe because it guarantees that every element of the
    /// slice will be initialized by consuming the return value of the closure.
    ///
    /// # Panics
    /// This method will panic if the closure `f` panics.
    #[must_use]
    pub fn init_each<F>(self, mut f: F) -> ArenaRef<[T]>
    where
        F: FnMut(usize) -> T,
    {
        let len = self.len();
        let ptr = self.ptr.as_ptr().cast::<MaybeUninit<T>>();

        for i in 0..len {
            // SAFETY: `ptr.add(i)` points to a valid, uninitialized `MaybeUninit<T>` element.
            // We are calling the safe `write` method on it.
            unsafe {
                (*ptr.add(i)).write(f(i));
            }
        }

        // SAFETY: All elements have been initialized in the loop above by calling `MaybeUninit::write`.
        // The `ptr` is valid for `len` elements, and `NonNull::slice_from_raw_parts` correctly
        // reinterprets the `MaybeUninit<T>` slice as a `T` slice because all elements are now initialized.
        unsafe {
            ArenaRef {
                ptr: NonNull::slice_from_raw_parts(NonNull::new_unchecked(ptr.cast::<T>()), len),
                generation: self.generation,
            }
        }
    }
}

impl<T: ?Sized> Deref for ArenaRef<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        unsafe { self.ptr.as_ref() }
    }
}

impl<T: ?Sized> DerefMut for ArenaRef<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.ptr.as_mut() }
    }
}

unsafe impl<T: ?Sized + Send> Send for ArenaRef<T> {}
unsafe impl<T: ?Sized + Sync> Sync for ArenaRef<T> {}
