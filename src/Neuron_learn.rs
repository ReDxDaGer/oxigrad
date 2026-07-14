struct Neuron{
    weights : Vec<f32>,
    bias : f32,
}

impl Neuron {
    fn new() -> Self {
        Self{
            weights: vec![0.2; 3], // Initialize with 3 weights
            bias: 0.0,
        }
    }
    fn forward(&self, inputs: &Vec<f32>) -> f32 {
        let mut sum = self.bias;
        for (weight, input) in self.weights.iter().zip(inputs.iter()) {
            sum += weight * input;
        }
        relu(sum)
    }
}

fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn tanh(x: f32) -> f32 {
    x.tanh()
}

fn leaky_relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.01 * x }
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
    return sum / n as f32
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
    let neuron = Neuron::new();
    let inputs = vec![222.0, 1.0, 34.0];
    let output = neuron.forward(&inputs);
    let loss = mseloss(&vec![output], &vec![50.0]);
    println!("Output: {}", output);
    println!("Loss: {}", loss);
}
