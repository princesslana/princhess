use dashmap::DashMap;
use nohash_hasher::BuildNoHashHasher;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::sync::{Arc, Mutex};

use crate::arena::{Arena, ArenaAllocator, ArenaError};
use crate::options::get_hash_size_mb;
use crate::search_tree::{HotMoveInfo, SearchNode};
use crate::state::State;

type Table = DashMap<u64, AtomicPtr<SearchNode>, BuildNoHashHasher<u64>>;

pub struct TranspositionTable {
    table: Table,
    #[allow(dead_code)]
    arena: Box<Arena>,
}

impl TranspositionTable {
    pub fn empty() -> Self {
        let table = Table::default();
        let arena = Box::new(Arena::new(get_hash_size_mb() / 2));

        Self::new(table, arena)
    }

    pub fn for_root() -> Self {
        Self::new(Table::default(), Box::new(Arena::new(2)))
    }

    fn new(table: Table, arena: Box<Arena>) -> Self {
        Self { table, arena }
    }

    pub fn arena(&self) -> &Arena {
        &self.arena
    }

    pub fn clear(&self) {
        self.table.clear();
        self.arena.clear();
    }

    pub fn insert<'a>(&'a self, key: &State, value: &'a SearchNode) -> Option<&'a SearchNode> {
        let hash = key.hash();
        if hash == 0 {
            return None;
        }

        if let Some(value_here) = self.table.get(&hash) {
            unsafe { Some(&*value_here.load(Ordering::Relaxed)) }
        } else {
            self.table
                .insert(hash, AtomicPtr::new(value as *const _ as *mut _));
            Some(value)
        }
    }

    pub fn lookup<'a>(&'a self, key: &State) -> Option<&'a SearchNode> {
        let hash = key.hash();

        self.table
            .get(&hash)
            .map(|v| unsafe { &*v.load(Ordering::Relaxed) })
    }

    pub fn lookup_into(&self, state: &State, dest: &mut SearchNode) {
        if let Some(src) = self.lookup(state) {
            dest.set_evaln(*src.evaln());

            let lhs = dest.hots();
            let rhs = src.hots();

            for i in 0..lhs.len().min(rhs.len()) {
                lhs[i].replace(&rhs[i]);
            }
        }
    }
}

pub struct LRTable {
    left: TranspositionTable,
    right: TranspositionTable,
    is_left_current: Arc<AtomicBool>,
    flip_lock: Mutex<()>,
}

impl LRTable {
    pub fn new(left: TranspositionTable, right: TranspositionTable) -> Self {
        Self {
            left,
            right,
            is_left_current: Arc::new(AtomicBool::new(true)),
            flip_lock: Mutex::new(()),
        }
    }

    pub fn is_arena_full(&self) -> bool {
        self.current_table().arena().full()
    }

    pub fn insert<'a>(&'a self, key: &State, value: &'a SearchNode) -> Option<&'a SearchNode> {
        self.current_table().insert(key, value)
    }

    pub fn lookup<'a>(&'a self, key: &State) -> Option<&'a SearchNode> {
        self.current_table().lookup(key)
    }

    pub fn lookup_into(&self, state: &State, dest: &mut SearchNode) {
        self.previous_table().lookup_into(state, dest);
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
            !self.is_left_current.load(Ordering::SeqCst),
            Ordering::SeqCst,
        );
    }

    pub fn table(self) -> TranspositionTable {
        if self.is_left_current() {
            self.left
        } else {
            self.right
        }
    }

    pub fn flip_lock(&self) -> &Mutex<()> {
        &self.flip_lock
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
    left: ArenaAllocator<'a>,
    right: ArenaAllocator<'a>,
}

impl<'a> LRAllocator<'a> {
    pub fn from_arenas(
        is_left_current: Arc<AtomicBool>,
        left: &'a Arena,
        right: &'a Arena,
    ) -> Self {
        Self {
            is_left_current,
            left: left.allocator(),
            right: right.allocator(),
        }
    }

    fn is_left_current(&self) -> bool {
        self.is_left_current.load(Ordering::Relaxed)
    }

    pub fn alloc_node(&self) -> Result<&'a mut SearchNode, ArenaError> {
        if self.is_left_current() {
            self.left.alloc_one()
        } else {
            self.right.alloc_one()
        }
    }

    pub fn alloc_move_info(&self, sz: usize) -> Result<&'a mut [HotMoveInfo], ArenaError> {
        if self.is_left_current() {
            self.left.alloc_slice(sz)
        } else {
            self.right.alloc_slice(sz)
        }
    }
}
