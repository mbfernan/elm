
/// Activation Functions to be used in the hidden layer.
#[derive(Clone)]
pub enum ActivationFunction {
    /// Leaky Rectified Linear Units
    ///
    /// # Papers:
    ///
    /// - (2013) [<https://ai.stanford.edu/%7Eamaas/papers/relu_hybrid_icml2013_final.pdf>]
    ///
    /// - (2015) [<https://arxiv.org/abs/1505.00853>]
    LeakyRelu
}


/// Map ActivationFunction enum variant to its function
pub fn map(activation_function: &ActivationFunction) -> fn(&mut f64) {
    match activation_function {
        ActivationFunction::LeakyRelu => { leaky_relu }
    }
}


/// [`Leaky Rectified Linear Units`]
///
/// ```
/// use elm::activation_functions::leaky_relu;
///
/// // When x >= 0
/// let mut x = 18.04;
/// leaky_relu(&mut x);
/// assert_eq!(x, 18.04);
///
/// // when x < 0.0
/// let mut x = -18.04;
/// leaky_relu(&mut x);
/// assert_eq!(x, -18.04 * 0.01);
/// ```
///
/// [`Leaky Rectified Linear Units`]: enum.ActivationFunction.html#variant.leakyrelu
pub fn leaky_relu(x: &mut f64) {
    if *x < 0f64 {
        *x *= 0.01;
    }
}
