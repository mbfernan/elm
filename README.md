# Extreme Learning Machines (ELMs)

A minimalistic crate that can be used to train ELMs (Neural Networks) for predicting regression tasks.

## Usage

```RUST
use elm::{ELM, Epsilon, Verbose};
use elm::activation_functions::ActivationFunction;

let mut elm = ELM::new(2, 4, 2, ActivationFunction::LeakyRelu, Epsilon::Default, Verbose::Quiet);
let inputs: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
let targets: Vec<Vec<f64>> = vec![vec![1.0, 1.0], vec![1.0, 1.5]];
elm.train(&inputs, &targets);

let new_inputs: Vec<Vec<f64>> = vec![vec![1.0, 4.0], vec![1.3, 0.6]];
let prediction = elm.predict(&new_inputs);
```