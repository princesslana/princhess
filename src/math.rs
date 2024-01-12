use std::time::{SystemTime, UNIX_EPOCH};

pub fn softmax(arr: &mut [f32], t: f32) {
    let max = max(arr);
    let mut s = 0.;

    for x in &mut *arr {
        *x = fastapprox::faster::exp((*x - max) / t);
        s += *x;
    }
    for x in &mut *arr {
        *x /= s;
    }
}

fn max(arr: &[f32]) -> f32 {
    let mut max = std::f32::NEG_INFINITY;
    for x in arr {
        max = max.max(*x);
    }
    max
}

pub struct Rng {
    seed: u64,
}

impl Rng {
    fn with_seed(seed: u64) -> Self {
        Self { seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.seed ^= self.seed << 13;
        self.seed ^= self.seed >> 17;
        self.seed ^= self.seed << 5;
        self.seed
    }

    pub fn next_usize(&mut self) -> usize {
        self.next_u64() as usize
    }

    pub fn next_i8(&mut self) -> i8 {
        self.next_u64() as i8
    }
}

impl Default for Rng {
    fn default() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();

        Self::with_seed(seed as u64)
    }
}
