fn count_neuron(
    inputs: Vec<f64>, neuron_weights: Vec<Vec<f64>>, neuron_biases: Vec<f64>
) -> Result<Vec<f64>, String> {
    if neuron_weights.len() != neuron_biases.len() {
        return Err("Lengths of weights and biases must be the same!".to_owned());
    }
    let mut layer_outputs = Vec::new();
    for (neuron_weight, neuron_bias) in neuron_weights.iter()
        .zip(neuron_biases.iter()) {
        if neuron_weight.len() != inputs.len() {
            return Err(
                format!(
                    "Cannot perform dot product on shape: [1:{:?}]:[{:?}:{:?}]",
                    inputs.len(),
                    neuron_weight.len(),
                    neuron_weights.len()
                )
            );
        }
        let mut neuron_output = 0.0;
        for (n_input, weight) in inputs.iter().zip(neuron_weight.iter()) {
            neuron_output += n_input * weight;
        }
        neuron_output += neuron_bias;
        layer_outputs.push(neuron_output);
    }
    return Ok(layer_outputs);
}

fn main() {
    let inputs = vec![1., 2., 3., 2.5];
    let weights = vec![
        vec![0.2, 0.8, -0.5, 1.],
        vec![0.5, -0.91, 0.26, -0.5],
        vec![-0.26, -0.27, 0.17, 0.87]
    ];
    let biases = vec![2., 3., 0.5];

    let output;
    match count_neuron(inputs, weights, biases) {
        Ok(data) => output = data,
        Err(e) => {
            println!("Error: {:?}", e);
            return ();
        }
    };
    println!("Output: {:?}", output);
}
