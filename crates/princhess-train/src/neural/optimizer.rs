use std::marker::PhantomData;

use crate::neural::lr_scheduler::LRScheduler;
use crate::neural::optimizable::Optimizable;

pub struct AdamWOptimizer<N: Optimizable, S: LRScheduler> {
    momentum: Vec<f32>,
    velocity: Vec<f32>,
    scheduler: S,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    step: u32,
    _phantom: PhantomData<N>,
}

impl<N: Optimizable, S: LRScheduler> AdamWOptimizer<N, S> {
    pub fn new(network: &N, scheduler: S) -> Self {
        let count = network.params().len();
        Self {
            momentum: vec![0.0; count],
            velocity: vec![0.0; count],
            scheduler,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            step: 0,
            _phantom: PhantomData,
        }
    }

    #[must_use]
    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn update(&mut self, network: &mut N, grad: &N) {
        self.step += 1;
        let bc1 = 1.0 - self.beta1.powi(self.step as i32);
        let bc2 = 1.0 - self.beta2.powi(self.step as i32);
        let lr = self.scheduler.get_lr(self.step);
        let (beta1, beta2, eps, wd) = (self.beta1, self.beta2, self.epsilon, self.weight_decay);

        for (((p, &g), m), v) in network
            .params_mut()
            .iter_mut()
            .zip(grad.params())
            .zip(self.momentum.iter_mut())
            .zip(self.velocity.iter_mut())
        {
            *m = beta1 * *m + (1.0 - beta1) * g;
            *v = beta2 * *v + (1.0 - beta2) * g * g;
            let m_hat = *m / bc1;
            let v_hat = *v / bc2;
            *p -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * *p);
        }
    }

    pub fn get_step(&self) -> u32 {
        self.step
    }

    pub fn get_learning_rate(&self) -> f32 {
        self.scheduler.get_lr(self.step)
    }
}
