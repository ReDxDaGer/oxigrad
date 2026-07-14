use std::cell::RefCell;
use std::collections::HashSet;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

// ============================================================
// Autograd engine
// ============================================================

/// A node in the computation graph. Cloning a Value clones the Rc (cheap,
/// shares the same underlying data/grad), which is what lets one Value
/// be a "child" of multiple parents in the graph.
#[derive(Clone)]
pub struct Value(Rc<RefCell<ValueData>>);

pub struct ValueData {
    pub data: f64,
    pub grad: f64,
    op: Op,
    children: Vec<Value>,
}

#[derive(Clone, Copy)]
enum Op {
    Leaf,
    Add,
    Mul,
    Pow(f64),
    Tanh,
    ReLU,
}

impl Value {
    pub fn new(data: f64) -> Self {
        Value(Rc::new(RefCell::new(ValueData {
            data,
            grad: 0.0,
            op: Op::Leaf,
            children: vec![],
        })))
    }

    fn from_op(data: f64, op: Op, children: Vec<Value>) -> Self {
        Value(Rc::new(RefCell::new(ValueData {
            data,
            grad: 0.0,
            op,
            children,
        })))
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    pub fn set_data(&self, d: f64) {
        self.0.borrow_mut().data = d;
    }

    pub fn zero_grad(&self) {
        self.0.borrow_mut().grad = 0.0;
    }

    fn add_grad(&self, g: f64) {
        self.0.borrow_mut().grad += g;
    }

    /// Pointer identity, used to dedup nodes during topological sort
    /// (the same Value can be a child of many parents).
    fn id(&self) -> usize {
        Rc::as_ptr(&self.0) as usize
    }

    pub fn pow(&self, n: f64) -> Value {
        let out = self.data().powf(n);
        Value::from_op(out, Op::Pow(n), vec![self.clone()])
    }

    pub fn tanh(&self) -> Value {
        let t = self.data().tanh();
        Value::from_op(t, Op::Tanh, vec![self.clone()])
    }

    pub fn relu(&self) -> Value {
        let d = self.data();
        let out = if d < 0.0 { 0.0 } else { d };
        Value::from_op(out, Op::ReLU, vec![self.clone()])
    }

    /// Applies the local chain rule for this node: given d(loss)/d(self),
    /// push the correctly-scaled gradient into this node's children.
    fn local_backward(&self) {
        let vd = self.0.borrow();
        let grad_out = vd.grad;
        match vd.op {
            Op::Leaf => {}
            Op::Add => {
                // d(a+b)/da = 1, d(a+b)/db = 1
                for c in &vd.children {
                    c.add_grad(grad_out);
                }
            }
            Op::Mul => {
                // d(a*b)/da = b, d(a*b)/db = a
                let a = &vd.children[0];
                let b = &vd.children[1];
                let a_data = a.data();
                let b_data = b.data();
                a.add_grad(b_data * grad_out);
                b.add_grad(a_data * grad_out);
            }
            Op::Pow(n) => {
                // d(a^n)/da = n * a^(n-1)
                let base = &vd.children[0];
                let base_data = base.data();
                base.add_grad(n * base_data.powf(n - 1.0) * grad_out);
            }
            Op::Tanh => {
                // d(tanh(a))/da = 1 - tanh(a)^2
                let child = &vd.children[0];
                child.add_grad((1.0 - vd.data * vd.data) * grad_out);
            }
            Op::ReLU => {
                // d(relu(a))/da = 1 if a > 0 else 0
                let child = &vd.children[0];
                let g = if vd.data > 0.0 { grad_out } else { 0.0 };
                child.add_grad(g);
            }
        }
    }

    /// Runs backprop from this node. Builds a topological order via DFS
    /// over the graph, then walks it in reverse, applying the chain rule
    /// at each node. Standard reverse-mode autodiff.
    pub fn backward(&self) {
        let mut topo: Vec<Value> = vec![];
        let mut visited: HashSet<usize> = HashSet::new();

        fn build(v: &Value, visited: &mut HashSet<usize>, topo: &mut Vec<Value>) {
            if visited.insert(v.id()) {
                for c in &v.0.borrow().children {
                    build(c, visited, topo);
                }
                topo.push(v.clone());
            }
        }
        build(self, &mut visited, &mut topo);

        self.0.borrow_mut().grad = 1.0; // d(self)/d(self) = 1
        for v in topo.iter().rev() {
            v.local_backward();
        }
    }
}

impl Add for Value {
    type Output = Value;
    fn add(self, other: Value) -> Value {
        let out = self.data() + other.data();
        Value::from_op(out, Op::Add, vec![self, other])
    }
}

impl Mul for Value {
    type Output = Value;
    fn mul(self, other: Value) -> Value {
        let out = self.data() * other.data();
        Value::from_op(out, Op::Mul, vec![self, other])
    }
}

impl Neg for Value {
    type Output = Value;
    fn neg(self) -> Value {
        self * Value::new(-1.0)
    }
}

impl Sub for Value {
    type Output = Value;
    fn sub(self, other: Value) -> Value {
        self + (-other)
    }
}

impl Div for Value {
    type Output = Value;
    fn div(self, other: Value) -> Value {
        self * other.pow(-1.0)
    }
}

// Convenience: allow `&Value + &Value` etc. without moving/cloning manually
// everywhere. Implemented by cloning the Rc (cheap).
impl Add for &Value {
    type Output = Value;
    fn add(self, other: &Value) -> Value {
        self.clone() + other.clone()
    }
}
impl Mul for &Value {
    type Output = Value;
    fn mul(self, other: &Value) -> Value {
        self.clone() * other.clone()
    }
}
impl Sub for &Value {
    type Output = Value;
    fn sub(self, other: &Value) -> Value {
        self.clone() - other.clone()
    }
}

// ============================================================
// Demo: train a single neuron with gradient descent
// ============================================================

// A single neuron: y = tanh(w1*x1 + w2*x2 + b)
struct Neuron {
    w1: Value,
    w2: Value,
    b: Value,
}

impl Neuron {
    fn new(w1: f64, w2: f64, b: f64) -> Self {
        Neuron {
            w1: Value::new(w1),
            w2: Value::new(w2),
            b: Value::new(b),
        }
    }

    fn forward(&self, x1: &Value, x2: &Value) -> Value {
        let sum = &(&self.w1 * x1) + &(&self.w2 * x2);
        let sum = &sum + &self.b;
        sum.tanh()
    }

    fn params(&self) -> [&Value; 3] {
        [&self.w1, &self.w2, &self.b]
    }
}

fn main() {
    // Toy dataset: 4 points, target is roughly "AND-ish" via tanh output
    let xs: [(f64, f64); 4] = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)];
    let ys: [f64; 4] = [1.0, -1.0, -1.0, -1.0];

    let n = Neuron::new(0.3, -0.2, 0.1);
    let lr = 0.05;
    let epochs = 200;

    for epoch in 0..epochs {
        // zero grads from previous step
        for p in n.params() {
            p.zero_grad();
        }

        // sum squared error loss over the dataset
        let mut loss = Value::new(0.0);
        for (x1, x2) in xs.iter() {
            let x1v = Value::new(*x1);
            let x2v = Value::new(*x2);
            let pred = n.forward(&x1v, &x2v);
            let target_idx = xs.iter().position(|p| p == &(*x1, *x2)).unwrap();
            let target = Value::new(ys[target_idx]);
            let diff = &pred - &target;
            loss = loss + diff.pow(2.0);
        }

        loss.backward();

        // gradient descent step
        for p in n.params() {
            let new_val = p.data() - lr * p.grad();
            p.set_data(new_val);
        }

        if epoch % 40 == 0 || epoch == epochs - 1 {
            println!("epoch {:>3}  loss = {:.6}", epoch, loss.data());
        }
    }

    println!(
        "\nfinal weights: w1={:.4} w2={:.4} b={:.4}",
        n.w1.data(),
        n.w2.data(),
        n.b.data()
    );
    println!("predictions:");
    for (x1, x2) in xs.iter() {
        let pred = n.forward(&Value::new(*x1), &Value::new(*x2));
        println!("  ({:>4}, {:>4}) -> {:.4}", x1, x2, pred.data());
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn test_add_mul_grad() {
        // f = a*b + b, a=2, b=-3
        // df/da = b = -3
        // df/db = a + 1 = 3
        let a = Value::new(2.0);
        let b = Value::new(-3.0);
        let f = a.clone() * b.clone() + b.clone();
        f.backward();
        assert!(approx(a.grad(), -3.0));
        assert!(approx(b.grad(), 3.0));
    }

    #[test]
    fn test_pow_and_div() {
        // f = a / b = a * b^-1, a=6, b=2 -> f=3
        // df/da = 1/b = 0.5
        // df/db = -a/b^2 = -1.5
        let a = Value::new(6.0);
        let b = Value::new(2.0);
        let f = a.clone() / b.clone();
        f.backward();
        assert!(approx(f.data(), 3.0));
        assert!(approx(a.grad(), 0.5));
        assert!(approx(b.grad(), -1.5));
    }

    #[test]
    fn test_shared_node_grad_accumulates() {
        // f = a + a  (a used twice) -> df/da = 2
        let a = Value::new(3.0);
        let f = a.clone() + a.clone();
        f.backward();
        assert!(approx(f.data(), 6.0));
        assert!(approx(a.grad(), 2.0));
    }

    #[test]
    fn test_tanh_grad() {
        // f = tanh(a), a=0 -> f=0, df/da = 1 - tanh(0)^2 = 1
        let a = Value::new(0.0);
        let f = a.tanh();
        f.backward();
        assert!(approx(f.data(), 0.0));
        assert!(approx(a.grad(), 1.0));
    }

    #[test]
    fn test_relu_grad() {
        let a = Value::new(-2.0);
        let f = a.relu();
        f.backward();
        assert!(approx(f.data(), 0.0));
        assert!(approx(a.grad(), 0.0));

        let a2 = Value::new(2.0);
        let f2 = a2.relu();
        f2.backward();
        assert!(approx(f2.data(), 2.0));
        assert!(approx(a2.grad(), 1.0));
    }
}
