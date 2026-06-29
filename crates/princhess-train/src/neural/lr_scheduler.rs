#[derive(Debug, Clone)]
pub struct LinearWarmupDecayLRScheduler {
    initial_lr: f32,
    warmup_steps: u32,
    total_steps: u32,
}

impl LinearWarmupDecayLRScheduler {
    #[must_use]
    pub fn new(initial_lr: f32, warmup_fraction: f32, total_steps: u32) -> Self {
        let warmup_steps = (total_steps as f32 * warmup_fraction).ceil() as u32;
        Self {
            initial_lr,
            warmup_steps,
            total_steps,
        }
    }
}

impl LRScheduler for LinearWarmupDecayLRScheduler {
    fn get_lr(&self, step: u32) -> f32 {
        if step < self.warmup_steps {
            self.initial_lr * step as f32 / self.warmup_steps as f32
        } else {
            let decay_steps = self.total_steps - self.warmup_steps;
            let elapsed = step - self.warmup_steps;
            self.initial_lr * (1.0 - elapsed as f32 / decay_steps as f32).max(0.0)
        }
    }
}

pub trait LRScheduler {
    fn get_lr(&self, step: u32) -> f32;
}

#[derive(Debug, Clone)]
pub struct StepLRScheduler {
    initial_lr: f32,
    drop_factor: f32,
    drop_interval: u32,
}

impl StepLRScheduler {
    #[must_use]
    pub fn new(initial_lr: f32, drop_factor: f32, drop_at_fraction: f32, total_steps: u32) -> Self {
        let drop_interval = (total_steps as f32 * drop_at_fraction).ceil() as u32;
        Self {
            initial_lr,
            drop_factor,
            drop_interval,
        }
    }
}

impl LRScheduler for StepLRScheduler {
    fn get_lr(&self, step: u32) -> f32 {
        let num_drops = step / self.drop_interval;
        self.initial_lr * self.drop_factor.powi(num_drops as i32)
    }
}

#[derive(Debug, Clone)]
pub struct ConstantLRScheduler {
    lr: f32,
}

impl ConstantLRScheduler {
    #[must_use]
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl LRScheduler for ConstantLRScheduler {
    fn get_lr(&self, _step: u32) -> f32 {
        self.lr
    }
}

#[derive(Debug, Clone)]
pub struct CosineAnnealingLRScheduler {
    initial_lr: f32,
    min_lr: f32,
    total_steps: u32,
    cycle_decay: f32,
}

impl CosineAnnealingLRScheduler {
    #[must_use]
    pub fn new(initial_lr: f32, min_lr: f32, total_steps: u32, cycle_decay: f32) -> Self {
        Self {
            initial_lr,
            min_lr,
            total_steps,
            cycle_decay,
        }
    }
}

impl LRScheduler for CosineAnnealingLRScheduler {
    fn get_lr(&self, step: u32) -> f32 {
        let cycle = step / self.total_steps;
        let cycle_step = step % self.total_steps;
        let peak_lr = self.initial_lr * self.cycle_decay.powi(cycle as i32);
        let progress = cycle_step as f32 / self.total_steps as f32;
        let cosine_factor = (1.0 + (std::f32::consts::PI * progress).cos()) * 0.5;
        self.min_lr + (peak_lr - self.min_lr) * cosine_factor
    }
}
