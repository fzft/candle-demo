use candle_core::{Tensor, Result, Device, D};
use candle_nn::VarBuilder;

// #[derive(Debug)]
// pub struct LayerNorm {
//     weight: Tensor,
//     bias: Tensor,
//     epsilon: f32,
// }

// impl LayerNorm {
//     pub fn load(vb: VarBuilder, hidden_size: usize, epsilon: f32) -> Result<Self> {
//         Ok(Self{
//             weight: vb
//             .get(hidden_size, "weight")?,
//             bias: vb.get(hidden_size, "bias")?,
//             epsilon,
//         })
//     }

//     pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
//         match hidden_states.device() {
//             Device::Cuda(_) => {
//                 let original_shape = hidden_states.shape().clone();
//                 let hidden_states = hidden_states.flatten_to(D::Minus2)?;
//                 let result = layer_norm(&hidden_states, &self.weight, Some(&self.bias), self.epsilon)?;
//                 result.reshape(&original_shape)?
//             },
//             _ => {unimplemented!("unsupported device")}

//         }
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Tensor, Device, D};

    #[test]
    fn test_layer_norm() {
        
    }
}