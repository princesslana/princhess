use std::alloc::{self, Layout};
use std::cell::UnsafeCell;
use std::mem::{self, MaybeUninit};
use std::ops::{Deref, DerefMut};
use std::ptr::{self, NonNull};
use std::slice;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use crate::mem::Align64;

#[derive(Debug)]
pub enum Error {
    Full,
}

const BYTES_PER_MB: usize = 1 << 20;
const CHUNK_SIZE_MB: usize = 2;
const CHUNK_SIZE: usize = BYTES_PER_MB * CHUNK_SIZE_MB;

type Chunk = Align64<[u8; CHUNK_SIZE]>;

// Global counter for arena generations, starting at 1.
// 0 is reserved for "invalid" allocators.
static GLOBAL_ARENA_GENERATION: AtomicU32 = AtomicU32::new(1);

pub struct Arena {
    chunks: UnsafeCell<Box<[Chunk]>>,
    max: usize,
    generation: AtomicU32,
    current_chunk_idx: AtomicUsize,
}

impl Arena {
    pub fn new(max_size_mb: usize) -> Self {
        let max = max_size_mb / CHUNK_SIZE_MB;

        let layout = Layout::array::<Chunk>(max).unwrap();
        let chunks_box = unsafe {
            let raw = alloc::alloc(layout);
            if raw.is_null() {
                alloc::handle_alloc_error(layout);
            }
            let slice = slice::from_raw_parts_mut(raw.cast(), max);
            Box::from_raw(slice)
        };

        Self {
            chunks: UnsafeCell::new(chunks_box),
            max,
            generation: AtomicU32::new(GLOBAL_ARENA_GENERATION.fetch_add(1, Ordering::AcqRel)),
            current_chunk_idx: AtomicUsize::new(0),
        }
    }

    fn used(&self) -> usize {
        self.current_chunk_idx.load(Ordering::Acquire).min(self.max)
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
        let idx = self.current_chunk_idx.fetch_add(1, Ordering::AcqRel);

        if idx >= self.max {
            return Err(Error::Full);
        }

        // SAFETY: We are accessing a specific chunk at `idx`.
        // The `current_chunk_idx` ensures that each chunk is allocated only once.
        // The `UnsafeCell` allows us to get a mutable pointer to the `Box<[Chunk]>`
        // from an immutable `&self`. We then dereference this to get a mutable
        // slice `&mut [Chunk]`, and then index into it to get `&mut Chunk`.
        // The `Align64` wrapper implements `DerefMut` to `[u8; CHUNK_SIZE]`,
        // which can then be converted to `&mut [u8]`.
        let chunk_ref: &mut Chunk = unsafe { &mut (*self.chunks.get())[idx] };
        Ok(&mut **chunk_ref)
    }

    pub fn allocator(&self) -> Result<Allocator<'_>, Error> {
        let current_generation = self.generation.load(Ordering::Acquire);
        let initial_chunk = self.alloc_chunk()?;
        Ok(Allocator {
            generation: AtomicU32::new(current_generation),
            arena: self,
            memory: UnsafeCell::new(Some(NonNull::from(initial_chunk))),
        })
    }

    pub fn stale_allocator(&self) -> Allocator<'_> {
        Allocator {
            // Set to 0 to ensure re-allocation on first use, as 0 is an invalid generation.
            generation: AtomicU32::new(0),
            arena: self,
            memory: UnsafeCell::new(None),
        }
    }

    pub fn clear(&self) {
        self.current_chunk_idx.store(0, Ordering::Release);
        self.generation.store(
            GLOBAL_ARENA_GENERATION.fetch_add(1, Ordering::AcqRel),
            Ordering::Release,
        );
    }

    #[must_use]
    pub fn generation(&self) -> u32 {
        self.generation.load(Ordering::Acquire)
    }
}

// This is safe because all mutations to `chunks` are synchronized via `current_chunk_idx`
// and `generation` atomics, ensuring no data races on the underlying memory.
unsafe impl Sync for Arena {}

pub struct Allocator<'a> {
    generation: AtomicU32,
    arena: &'a Arena,
    memory: UnsafeCell<Option<NonNull<[u8]>>>,
}

const ALIGN: usize = 8;

impl<'a> Allocator<'a> {
    fn get_memory(&self, sz: usize) -> Result<&'a mut [u8], Error> {
        // A single allocation request should never be larger than a chunk.
        assert!(sz <= CHUNK_SIZE, "Allocation request exceeds CHUNK_SIZE");

        let current_arena_generation = self.arena.generation.load(Ordering::Acquire);
        let allocator_generation = self.generation.load(Ordering::Acquire);

        let current_memory_slot = unsafe { &mut *self.memory.get() };

        let mut chunk_to_use_ptr: NonNull<[u8]>;

        // Condition 1: Allocator is stale (generation mismatch) or has no chunk yet (None)
        // This covers the initial state of `stale_allocator` (generation 0, memory None)
        // and any allocator whose arena has been cleared.
        if allocator_generation != current_arena_generation || current_memory_slot.is_none() {
            let new_slice = self.arena.alloc_chunk()?;
            chunk_to_use_ptr = NonNull::from(new_slice);
            *current_memory_slot = Some(chunk_to_use_ptr);
            self.generation
                .store(current_arena_generation, Ordering::Release);
        } else {
            // Condition 2: Allocator is current, check if current chunk has enough space
            let existing_chunk_ptr = current_memory_slot
                .expect("Allocator memory slot should be initialized and not None at this point");
            let existing_slice_len = unsafe { existing_chunk_ptr.as_ref().len() };

            if sz <= existing_slice_len {
                chunk_to_use_ptr = existing_chunk_ptr;
            } else {
                let new_slice = self.arena.alloc_chunk()?;
                chunk_to_use_ptr = NonNull::from(new_slice);
                *current_memory_slot = Some(chunk_to_use_ptr);
                // Generation is already current, no need to update
            }
        }

        // Now, `chunk_to_use_ptr` holds the NonNull pointer to the slice we will use.
        // It's guaranteed to be a valid pointer to a slice of at least `sz` length
        // (because `alloc_chunk` returns a `CHUNK_SIZE` slice, and `sz <= CHUNK_SIZE` is checked).
        let slice_to_return = unsafe { chunk_to_use_ptr.as_mut() };
        let (left, right) = slice_to_return.split_at_mut(sz);

        // Update the remaining part of the slice in the Allocator's memory slot
        *current_memory_slot = Some(NonNull::from(right));
        Ok(left)
    }

    pub fn alloc_one<T>(&self) -> Result<ArenaRef<MaybeUninit<T>>, Error> {
        assert!(ALIGN.is_multiple_of(mem::align_of::<T>()));
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
        assert!(ALIGN.is_multiple_of(mem::align_of::<T>()));
        let x = mem::size_of::<T>();
        let x = (x + ALIGN - 1) & !(ALIGN - 1);
        let x = self.get_memory(x * sz)?;
        let current_generation = self.arena.generation.load(Ordering::Acquire);
        Ok(ArenaRef {
            ptr: unsafe {
                NonNull::new_unchecked(ptr::slice_from_raw_parts_mut(
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
    generation: u32,
}

impl<T: ?Sized> ArenaRef<T> {
    #[must_use]
    pub fn from_raw_parts(ptr: NonNull<T>, generation: u32) -> Self {
        Self { ptr, generation }
    }

    #[must_use]
    pub fn into_non_null(self) -> NonNull<T> {
        self.ptr
    }

    #[must_use]
    pub fn generation(&self) -> u32 {
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
