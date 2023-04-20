use dashmap::DashMap;
use nohash_hasher::BuildNoHashHasher;
use std::sync::atomic::{AtomicPtr, Ordering};

use crate::arena::Arena;
use crate::options::get_hash_size_mb;
use crate::search_tree::{NodeStats, SearchNode};
use crate::state::State;

pub trait TranspositionHash {
    fn hash(&self) -> u64;
}

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
            dest.replace(src);
            dest.set_evaln(*src.evaln());

            let lhs = dest.hots();
            let rhs = src.hots();

            for i in 0..lhs.len().min(rhs.len()) {
                lhs[i].replace(&rhs[i]);
            }
        }
    }
}
