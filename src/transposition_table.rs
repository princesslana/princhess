use nohash_hasher::BuildNoHashHasher;
use scc::hash_map::HashMap;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::sync::{Arc, Mutex};

use crate::arena::{Allocator, Arena, Error as ArenaError};
use crate::search_tree::{MoveEdge, PositionNode};
use crate::state::State;

type Table = HashMap<u64, AtomicPtr<PositionNode>, BuildNoHashHasher<u64>>;

pub struct TranspositionTable {
    table: Table,
    arena: Box<Arena>,
}

impl TranspositionTable {
    #[must_use]
    pub fn empty(hash_size_mb: usize) -> Self {
        let table = Table::default();
        let arena = Box::new(Arena::new(hash_size_mb));

        Self::new(table, arena)
    }

    #[must_use]
    pub fn for_root() -> Self {
        Self::new(Table::default(), Box::new(Arena::new(2)))
    }

    fn new(table: Table, arena: Box<Arena>) -> Self {
        Self { table, arena }
    }

    #[must_use]
    pub fn full(&self) -> usize {
        self.arena.full()
    }

    #[must_use]
    pub fn arena(&self) -> &Arena {
        &self.arena
    }

    pub fn clear(&self) {
        self.table.clear();
        self.arena.clear();
    }

    #[must_use]
    pub fn insert<'a>(&'a self, key: &State, value: &'a PositionNode) -> &'a PositionNode {
        let hash = key.hash();

        match self
            .table
            .insert(hash, AtomicPtr::new(ptr::from_ref(value).cast_mut()))
        {
            Ok(()) => value,
            Err(_) => self
                .table
                .read(&hash, |_, v| unsafe { &*v.load(Ordering::Relaxed) })
                .unwrap(),
        }
    }

    #[must_use]
    pub fn lookup<'a>(&'a self, key: &State) -> Option<&'a PositionNode> {
        let hash = key.hash();

        self.table
            .read(&hash, |_, v| unsafe { &*v.load(Ordering::Relaxed) })
    }

    pub fn lookup_into(&self, state: &State, dest: &mut PositionNode) -> bool {
        if let Some(src) = self.lookup(state) {
            dest.set_flag(src.flag());

            let lhs = dest.hots();
            let rhs = src.hots();

            for i in 0..lhs.len().min(rhs.len()) {
                lhs[i].replace(&rhs[i]);
            }
            true
        } else {
            false
        }
    }
}

pub struct LRTable {
    left: TranspositionTable,
    right: TranspositionTable,
    is_left_current: Arc<AtomicBool>,
    is_flipping: AtomicBool,
    flip_lock: Mutex<()>,
}

impl LRTable {
    #[must_use]
    pub fn new(left: TranspositionTable, right: TranspositionTable) -> Self {
        Self {
            left,
            right,
            is_left_current: Arc::new(AtomicBool::new(true)),
            is_flipping: AtomicBool::new(false),
            flip_lock: Mutex::new(()),
        }
    }

    #[must_use]
    pub fn empty(hash_size_mb: usize) -> Self {
        let each_size = hash_size_mb / 2;
        Self::new(
            TranspositionTable::empty(each_size),
            TranspositionTable::empty(each_size),
        )
    }

    pub fn full(&self) -> usize {
        (self.left.arena().full() + self.right.arena().full()) / 2
    }

    pub fn capacity_remaining(&self) -> usize {
        self.current_table().arena().capacity_remaining()
    }

    pub fn insert<'a>(&'a self, key: &State, value: &'a PositionNode) -> &'a PositionNode {
        self.current_table().insert(key, value)
    }

    pub fn lookup<'a>(&'a self, key: &State) -> Option<&'a PositionNode> {
        self.current_table().lookup(key)
    }

    pub fn lookup_into(&self, state: &State, dest: &mut PositionNode) {
        self.previous_table().lookup_into(state, dest);
    }

    pub fn lookup_into_from_all(&self, state: &State, dest: &mut PositionNode) {
        if !self.current_table().lookup_into(state, dest) {
            self.previous_table().lookup_into(state, dest);
        }
    }

    pub fn is_left_current(&self) -> bool {
        self.is_left_current.load(Ordering::Relaxed)
    }

    fn current_table(&self) -> &TranspositionTable {
        if self.is_left_current() {
            &self.left
        } else {
            &self.right
        }
    }

    fn previous_table(&self) -> &TranspositionTable {
        if self.is_left_current() {
            &self.right
        } else {
            &self.left
        }
    }

    pub fn flip_tables(&self) {
        self.previous_table().clear();
        self.is_left_current.store(
            !self.is_left_current.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }

    pub fn wait_if_flipping(&self) {
        if self.is_flipping.load(Ordering::Relaxed) {
            let _lock = self.flip_lock.lock().unwrap();
        }
    }

    pub fn flip_if_full<F>(&self, f: F)
    where
        F: FnOnce(),
    {
        self.is_flipping.store(true, Ordering::Relaxed);

        {
            let _lock = self.flip_lock.lock().unwrap();

            if self.current_table().arena().is_full() {
                self.flip_tables();
                f();
            }
        }

        self.is_flipping.store(false, Ordering::Relaxed);
    }

    pub fn allocator(&self) -> LRAllocator {
        LRAllocator::from_arenas(
            self.is_left_current.clone(),
            self.left.arena(),
            self.right.arena(),
        )
    }
}

pub struct LRAllocator<'a> {
    is_left_current: Arc<AtomicBool>,
    left: Allocator<'a>,
    right: Allocator<'a>,
}

impl<'a> LRAllocator<'a> {
    pub fn from_arenas(
        is_left_current: Arc<AtomicBool>,
        left: &'a Arena,
        right: &'a Arena,
    ) -> Self {
        Self {
            is_left_current,
            left: left.invalid_allocator(),
            right: right.invalid_allocator(),
        }
    }

    fn is_left_current(&self) -> bool {
        self.is_left_current.load(Ordering::Relaxed)
    }

    pub fn alloc_node(&self) -> Result<&'a mut PositionNode, ArenaError> {
        if self.is_left_current() {
            self.left.alloc_one()
        } else {
            self.right.alloc_one()
        }
    }

    pub fn alloc_move_info(&self, sz: usize) -> Result<&'a mut [MoveEdge], ArenaError> {
        if self.is_left_current() {
            self.left.alloc_slice(sz)
        } else {
            self.right.alloc_slice(sz)
        }
    }
}
