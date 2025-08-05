pub trait OutputLayer<OutputType> {
    fn output_layer(&self) -> OutputType;
}

use crate::neural::lr_scheduler::LRScheduler;
use crate::neural::optimizer::AdamWOptimizer;

pub trait FeedForwardNetwork: Sized {
    type InputType: Clone;
    type OutputType: Clone;
    type Layers: OutputLayer<Self::OutputType>;

    fn adamw<S: LRScheduler>(
        &mut self,
        g: &Self,
        m: &mut Self,
        v: &mut Self,
        optimizer: &AdamWOptimizer<S>,
    );

    fn out_with_layers(&self, input: &Self::InputType) -> Self::Layers;

    fn out(&self, input: &Self::InputType) -> Self::OutputType {
        self.out_with_layers(input).output_layer()
    }

    fn backprop(
        &self,
        input: &Self::InputType,
        grad: &mut Self,
        out_err: Self::OutputType,
        layers: &Self::Layers,
    ) -> Self::InputType;
}
