# Extreme Learning Machines (ELMs)

Extreme Learning Machine (ELM) crate. A minimalistic and flexible crate that can be used to train ELMs,
a type of Neural Networks. Currently supports a single hidden layer and regression tasks.

References:

- Original paper: <https://ieeexplore.ieee.org/document/1380068>

- Wikipedia: <https://en.wikipedia.org/wiki/Extreme_learning_machine>


## Basic usage

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