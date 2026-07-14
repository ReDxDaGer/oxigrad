#[derive(Clone, Copy)]
enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    LeakyReLU,
}

impl Activation {
    fn apply(&self, x: f32) -> f32 {
        match self {
            Activation::ReLU => relu(x),
            Activation::Sigmoid => sigmoid(x),
            Activation::Tanh => tanh(x),
            Activation::LeakyReLU => leaky_relu(x),
        }
    }
}

struct Neuron {
    weights: Vec<f32>,
    bias: f32,
    activation: Activation,
}

impl Neuron {
    fn new(n_inputs: usize, activation: Activation) -> Self {
        Self {
            weights: vec![0.2; n_inputs],
            bias: 0.0,
            activation,
        }
    }

    fn random_ish(n_inputs: usize, seed: usize, activation: Activation) -> Self {
        let weights = (0..n_inputs)
            .map(|i| ((seed + i * 7) % 11) as f32 / 10.0 - 0.5)
            .collect();
        Self {
            weights,
            bias: 0.05,
            activation,
        }
    }

    fn forward(&self, inputs: &Vec<f32>) -> f32 {
        let mut sum = self.bias;
        for (weight, input) in self.weights.iter().zip(inputs.iter()) {
            sum += weight * input;
        }
        self.activation.apply(sum)
    }

    fn nudge(&mut self, idx: usize, delta: f32) {
        if let Some(w) = self.weights.get_mut(idx) {
            *w += delta;
        }
    }
}

struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn new(n_neurons: usize, n_inputs: usize, activation: Activation) -> Self {
        let neurons = (0..n_neurons)
            .map(|i| Neuron::random_ish(n_inputs, i * 3 + 1, activation))
            .collect();
        Self { neurons }
    }

    fn forward(&self, inputs: &Vec<f32>) -> Vec<f32> {
        self.neurons.iter().map(|n| n.forward(inputs)).collect()
    }
}

struct Network {
    layers: Vec<Layer>,
}

impl Network {
    fn new(layer_sizes: &[usize], input_size: usize, activation: Activation) -> Self {
        let mut layers = Vec::new();
        let mut prev = input_size;
        for &size in layer_sizes {
            layers.push(Layer::new(size, prev, activation));
            prev = size;
        }
        Self { layers }
    }

    fn forward(&self, inputs: &Vec<f32>) -> Vec<f32> {
        let mut out = inputs.clone();
        for layer in &self.layers {
            out = layer.forward(&out);
        }
        out
    }
}

fn relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn tanh(x: f32) -> f32 {
    x.tanh()
}

fn leaky_relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.01 * x
    }
}

fn softmax(inputs: &Vec<f32>) -> Vec<f32> {
    let max_input = inputs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_values: Vec<f32> = inputs.iter().map(|&x| (x - max_input).exp()).collect();
    let sum_exp_values: f32 = exp_values.iter().sum();
    exp_values.iter().map(|&x| x / sum_exp_values).collect()
}

fn mseloss(predictions: &Vec<f32>, targets: &Vec<f32>) -> f32 {
    let n = predictions.len();
    let mut sum = 0.0;
    for (pred, target) in predictions.iter().zip(targets.iter()) {
        sum += (pred - target).powi(2);
    }
    sum / n as f32
}

fn cross_entropy_loss(predictions: &Vec<f64>, targets: &Vec<f64>) -> f64 {
    let n = predictions.len();
    let mut sum = 0.0;
    for (pred, target) in predictions.iter().zip(targets.iter()) {
        sum += -target * pred.ln();
    }
    sum / n as f64
}

fn binary_cross_entropy_loss(predictions: &Vec<f64>, targets: &Vec<f64>) -> f64 {
    let n = predictions.len();
    let mut sum = 0.0;
    for (pred, target) in predictions.iter().zip(targets.iter()) {
        sum += -target * pred.ln() - (1.0 - target) * (1.0 - pred).ln();
    }
    sum / n as f64
}

fn main() {
    println!("--- single neuron, same as before ---");
    let neuron = Neuron::new(3, Activation::ReLU);
    let inputs = vec![222.0, 1.0, 34.0];
    let output = neuron.forward(&inputs);
    let loss = mseloss(&vec![output], &vec![50.0]);
    println!("Output: {}", output);
    println!("Loss: {}", loss);

    println!("\n--- ok that blew up instantly because inputs are huge, normalizing ---");
    let inputs_norm = vec![0.9, 0.05, 0.4];
    let out_norm = neuron.forward(&inputs_norm);
    println!("Output (normalized inputs): {}", out_norm);

    println!("\n--- trying the other activations on the same weighted sum ---");
    for act in [
        Activation::ReLU,
        Activation::Sigmoid,
        Activation::Tanh,
        Activation::LeakyReLU,
    ] {
        let n = Neuron {
            weights: neuron.weights.clone(),
            bias: neuron.bias,
            activation: act,
        };
        println!("Output: {}", n.forward(&inputs_norm));
    }

    println!("\n--- one neuron isn't a network, making a layer ---");
    let hidden_layer = Layer::new(4, 3, Activation::ReLU);
    let hidden_out = hidden_layer.forward(&inputs_norm);
    println!("Hidden layer outputs: {:?}", hidden_out);

    println!("\n--- stacking layers into an actual network ---");
    let net = Network::new(&[4, 4, 3], 3, Activation::Tanh);
    let net_out = net.forward(&inputs_norm);
    println!("Network raw output: {:?}", net_out);

    println!("\n--- softmax on the final layer since it looks like a classifier now ---");
    let probs = softmax(&net_out);
    println!("Softmax output: {:?}", probs);
    println!("Sums to: {}", probs.iter().sum::<f32>());

    println!("\n--- manually tweaking one neuron's weights and watching loss move ---");
    let mut tweak_neuron = Neuron::new(3, Activation::Sigmoid);
    let target = vec![1.0];
    for step in 0..6 {
        let out = tweak_neuron.forward(&inputs_norm);
        let l = mseloss(&vec![out], &target);
        println!("step {} -> output {:.4}, loss {:.4}", step, out, l);
        tweak_neuron.nudge(0, 0.1);
        tweak_neuron.nudge(1, 0.1);
        tweak_neuron.nudge(2, 0.1);
    }

    println!("\n--- checking the other losses actually run ---");
    let ce_preds = vec![0.7, 0.2, 0.1];
    let ce_targets = vec![1.0, 0.0, 0.0];
    println!("cross entropy: {}", cross_entropy_loss(&ce_preds, &ce_targets));

    let bce_preds = vec![0.8];
    let bce_targets = vec![1.0];
    println!(
        "binary cross entropy: {}",
        binary_cross_entropy_loss(&bce_preds, &bce_targets)
    );
}
