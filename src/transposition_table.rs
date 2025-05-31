use nohash_hasher::BuildNoHashHasher;
use scc::hash_map::HashMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::arena::{Allocator, Arena, ArenaRef, Error as ArenaError};
use crate::search_tree::{MoveEdge, PositionNode};
use crate::state::State;

struct Entry {
    node_ptr: NonNull<PositionNode>,
    generation: u64,
}

unsafe impl Send for Entry {}
unsafe impl Sync for Entry {}

impl Entry {
    fn new(node_ptr: NonNull<PositionNode>, generation: u64) -> Self {
        Self {
            node_ptr,
            generation,
        }
    }
}

type Table = HashMap<u64, Entry, BuildNoHashHasher<u64>>;

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
    pub fn insert<'a>(&'a self, key: &State, value: ArenaRef<PositionNode>) -> &'a PositionNode {
        let hash = key.hash();
        let generation = value.generation();
        let node_ptr = value.into_non_null();

        match self.table.insert(hash, Entry::new(node_ptr, generation)) {
            Ok(()) => unsafe { node_ptr.as_ref() },
            Err(_) => self
                .table
                .read(&hash, |_, entry| unsafe { entry.node_ptr.as_ref() })
                .unwrap(),
        }
    }

    #[must_use]
    pub fn lookup<'a>(&'a self, key: &State) -> Option<&'a PositionNode> {
        let hash = key.hash();

        self.table.read(&hash, |_, entry| {
            let current_arena_gen = self.arena.generation();

            if entry.generation == current_arena_gen {
                unsafe { Some(entry.node_ptr.as_ref()) }
            } else {
                None
            }
        })?
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
    tables: [TranspositionTable; 2],
    current: Arc<AtomicBool>,
    flip_lock: Mutex<()>,
}

impl LRTable {
    #[must_use]
    pub fn new(left: TranspositionTable, right: TranspositionTable) -> Self {
        Self {
            tables: [left, right],
            current: Arc::new(AtomicBool::new(false)),
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
        self.tables
            .iter()
            .map(TranspositionTable::full)
            .sum::<usize>()
            / self.tables.len()
    }

    pub fn capacity_remaining(&self) -> usize {
        self.current_table().arena().capacity_remaining()
    }

    pub fn insert<'a>(&'a self, key: &State, value: ArenaRef<PositionNode>) -> &'a PositionNode {
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

    fn current_table(&self) -> &TranspositionTable {
        &self.tables[usize::from(self.current.load(Ordering::Acquire))]
    }

    fn previous_table(&self) -> &TranspositionTable {
        &self.tables[usize::from(!self.current.load(Ordering::Acquire))]
    }

    fn flip_tables(&self) {
        self.previous_table().clear();
        self.current.fetch_not(Ordering::Release);
    }

    pub fn flip<F>(&self, f: F)
    where
        F: FnOnce(),
    {
        let _lock = self.flip_lock.lock().unwrap();
        self.flip_tables();
        f();
    }

    pub fn flip_if_full<F>(&self, f: F)
    where
        F: FnOnce(),
    {
        let _lock = self.flip_lock.lock().unwrap();

        if self.current_table().arena().is_full() {
            self.flip_tables();
            f();
        }
    }

    pub fn allocator(&self) -> LRAllocator {
        LRAllocator::from_arenas(
            self.current.clone(),
            self.tables[0].arena(),
            self.tables[1].arena(),
        )
    }
}

pub struct LRAllocator<'a> {
    allocators: [Allocator<'a>; 2],
    current: Arc<AtomicBool>,
}

impl<'a> LRAllocator<'a> {
    pub fn from_arenas(current: Arc<AtomicBool>, left: &'a Arena, right: &'a Arena) -> Self {
        Self {
            allocators: [left.invalid_allocator(), right.invalid_allocator()],
            current,
        }
    }
    pub fn alloc_node(
        &self,
        edges: usize,
    ) -> Result<(ArenaRef<PositionNode>, ArenaRef<[MoveEdge]>), ArenaError> {
        let alloc = &self.allocators[usize::from(self.current.load(Ordering::Acquire))];

        let edges_ref = alloc.alloc_slice(edges)?;
        let node_ref = alloc.alloc_one()?;

        Ok((node_ref, edges_ref))
    }
}
