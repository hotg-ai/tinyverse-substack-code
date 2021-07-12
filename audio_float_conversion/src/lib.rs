#![no_std]

extern crate alloc;

use rune_core::{Tensor};
use rune_proc_blocks::{ProcBlock, Transform};


#[derive(Debug, Clone, PartialEq, ProcBlock)]

#[transform(input = [i16; _], output = [f32; _])]

pub struct AudioFloatConversion {

    i16_max_as_float: f32,

}

const I16_MAX_AS_FLOAT: f32 = i16::MAX as f32;

impl AudioFloatConversion {

    pub const fn new() -> Self {

        AudioFloatConversion {

            i16_max_as_float: I16_MAX_AS_FLOAT,

        }

    }

}

impl Default for AudioFloatConversion {

    fn default() -> Self { AudioFloatConversion::new() }

}

impl Transform<Tensor<i16>> for AudioFloatConversion {

    type Output = Tensor<f32>;

    fn transform(&mut self, input: Tensor<i16>) -> Self::Output {

        input.map(|_dims, &value| (value as f32 / i16::MAX as f32).clamp(-1.0, 1.0))

    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    #[test]

    fn handle_empty() {
        let mut pb = AudioFloatConversion::new();

        let input = Tensor::new_vector(vec![0;15]);
        let got = pb.transform(input);
        assert_eq!(got.dimensions(), &[15]);
    }

    #[test]

    fn does_it_match() {

        let max = i16::MAX;

        let min = i16::MIN;

        let mut pb = AudioFloatConversion::new();

        let input = Tensor::new_vector(vec![0, max, min+1]);

        let got = pb.transform(input);

        assert_eq!(got.elements()[0..3], [0.0, 1.0, -1.0]);

    }

    #[test]

    fn does_clutch_work() {

        let max = i16::MAX;

        let min = i16::MIN;

        let mut pb = AudioFloatConversion::new();

        let input = Tensor::new_vector(vec![max, min, min+1]);

        let got = pb.transform(input);

        assert_eq!(got.elements()[0..3], [1.0, -1.0, -1.0]);

    }
    
}
