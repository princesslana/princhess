pub trait OutputLayer<OutputType> {
    fn output_layer(&self) -> OutputType;
}

pub trait FeedForwardNetwork: Sized {
    type InputType: Clone;
    type OutputType: Clone;
    type Layers: OutputLayer<Self::OutputType>;

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
