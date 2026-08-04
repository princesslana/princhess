use std::time::{SystemTime, UNIX_EPOCH};

/// Calculate Gini impurity from a distribution of counts
#[inline]
pub fn gini<I>(counts: I, total: u64) -> f32
where
    I: IntoIterator<Item = u32>,
{
    if total == 0 {
        return 0.0;
    }

    let mut sum_squares = 0.0_f32;
    for count in counts {
        let v = count as f32;
        sum_squares += v * v;
    }

    let total_f = total as f32;
    (1.0 - (sum_squares / (total_f * total_f))).clamp(0.0, 1.0)
}

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
    let mut max = f32::NEG_INFINITY;
    for x in arr {
        max = max.max(*x);
    }
    max
}

#[must_use]
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

    pub fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    pub fn next_usize(&mut self) -> usize {
        self.next_u64() as usize
    }

    // Returns a random f32 in the range [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1 << 24) as f32
    }

    pub fn next_f32_range(&mut self, min: f32, max: f32) -> f32 {
        min + self.next_f32() * (max - min)
    }

    fn next_normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(f32::EPSILON);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    #[allow(clippy::many_single_char_names)]
    fn next_gamma(&mut self, alpha: f32) -> f32 {
        if alpha < 1.0 {
            return self.next_gamma(alpha + 1.0) * self.next_f32().powf(1.0 / alpha);
        }
        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let z = self.next_normal();
            let v_base = 1.0 + c * z;
            if v_base <= 0.0 {
                continue;
            }
            let v = v_base * v_base * v_base;
            let u = self.next_f32().max(f32::EPSILON);
            if u < 1.0 - 0.0331 * (z * z) * (z * z) {
                return d * v;
            }
            if u.ln() < 0.5 * z * z + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }

    pub fn fill_dirichlet(&mut self, alpha: f32, out: &mut [f32]) {
        let mut sum = 0.0;
        for x in out.iter_mut() {
            *x = self.next_gamma(alpha);
            sum += *x;
        }
        for x in out.iter_mut() {
            *x /= sum;
        }
    }

    pub fn weighted(&mut self, weights: &[f32]) -> usize {
        let r = self.next_f32();

        let mut cumulative = 0.;

        for (i, &w) in weights.iter().enumerate() {
            cumulative += w;
            if r < cumulative {
                return i;
            }
        }

        weights.len() - 1
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
