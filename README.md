# Rust-autograd

A Rust port of [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd) — a scalar-valued, reverse-mode automatic differentiation engine, in one file, with zero dependencies beyond the Rust standard library.

Like the original, this is not built for speed or scale. It's built so that every gradient the engine computes can be traced back to a single line of code that implements one derivative rule. If you understand this file, you understand how backprop works in PyTorch, TensorFlow, and every other framework — they just do the same thing with tensors instead of scalars, and with a lot more engineering around it.

## What's actually in here

Everything lives in `src/main.rs`:

1. **The engine** — a `Value` type representing one node in a computation graph, with operators (`+`, `-`, `*`, `/`, `.pow()`, `.tanh()`, `.relu()`) that build the graph as you compute, and a `.backward()` method that runs reverse-mode autodiff over it.
2. **A demo** — a single neuron (`tanh(w1*x1 + w2*x2 + b)`) trained with plain gradient descent on 4 data points, so you can see the engine actually learn something.
3. **Tests** — 5 unit tests that check gradients against hand-derived calculus, so the math is verifiably correct, not just "looks right."

## How the engine works

### The graph

```rust
pub struct Value(Rc<RefCell<ValueData>>);

pub struct ValueData {
    pub data: f64,      // the forward-pass value
    pub grad: f64,       // d(loss)/d(this node), filled in by backward()
    op: Op,               // what operation produced this node
    children: Vec<Value>, // the inputs to that operation
}
```

Every arithmetic operation doesn't just compute a number — it also records *how* that number was computed, by wrapping it in a new `Value` that points back to its inputs (`children`) and the `Op` that combined them. Do enough arithmetic and you've built a DAG (directed acyclic graph) without ever calling a "build graph" function — the graph is a side effect of doing the math.

`Value` is `Rc<RefCell<...>>` because a single node can be an input to *multiple* later computations (e.g. `let y = &x * &x;` uses `x` twice), so the graph needs shared ownership. `RefCell` gives interior mutability so `.grad` can be updated without every `Value` needing to be `mut`.

### Forward pass

There isn't a separate "forward pass" step — it happens automatically as you write normal-looking Rust expressions:

```rust
let x1 = Value::new(1.0);
let w1 = Value::new(0.5);
let y = (&w1 * &x1).tanh();   // graph built here, y.data() already computed
```

### Backward pass

```rust
pub fn backward(&self) {
    // 1. Topologically sort the graph via DFS, so every node comes
    //    after all the nodes that depend on it.
    // 2. Seed this node's own gradient as 1.0 (d(y)/d(y) = 1).
    // 3. Walk the sorted list in reverse, and at each node, push its
    //    gradient into its children according to the chain rule.
}
```

Each `Op` variant has one local derivative rule in `local_backward()`:

| Op | Forward | Local gradient pushed to children |
|---|---|---|
| `Add` | `a + b` | `da += grad`, `db += grad` |
| `Mul` | `a * b` | `da += b.data * grad`, `db += a.data * grad` |
| `Pow(n)` | `a.powf(n)` | `da += n * a.data.powf(n-1) * grad` |
| `Tanh` | `a.tanh()` | `da += (1 - out^2) * grad` |
| `ReLU` | `max(0, a)` | `da += grad if a > 0 else 0` |

Gradients are **accumulated** (`+=`), not overwritten — this is what makes reusing a `Value` multiple times (like `x * x`) produce the correct summed gradient instead of silently corrupting it.

Subtraction and division aren't separate ops — `a - b` is defined as `a + (-b)`, and `a / b` as `a * b.pow(-1.0)`. Fewer cases to get wrong.

## Running it

```bash
cargo test     # runs the 5 gradient-correctness tests
cargo run      # trains the demo neuron and prints the loss curve
```

Expected test output:
```
running 5 tests
test tests::test_add_mul_grad ... ok
test tests::test_pow_and_div ... ok
test tests::test_relu_grad ... ok
test tests::test_shared_node_grad_accumulates ... ok
test tests::test_tanh_grad ... ok
```

Expected demo output (abbreviated):
```
epoch   0  loss = 4.391191
epoch  40  loss = 0.124409
...
epoch 199  loss = 0.021902
```

## How this differs from Python micrograd

| | Python micrograd | this |
|---|---|---|
| Graph ownership | Python's GC handles shared references for free | explicit `Rc<RefCell<>>` |
| Backward dispatch | each `Value` stores a `_backward` closure set at creation | a single `match` on an `Op` enum, dispatched at backward time |
| Type safety | none — anything can go in a `Value` | operator misuse (e.g. wrong arity) is a compile error |
| Performance | slow, dynamic | faster, but still scalar (one `f64` per node — no SIMD, no batching) |

The closures-per-node approach from the Python original is idiomatic there but awkward in Rust (closures capturing `Rc` clones work, but add indirection for no real benefit at this scale) — an enum + match is simpler and just as correct, which is why this port structures it that way instead of translating literally.

## Known limitations (by design)

- **Scalar only.** Every number is one `f64` `Value`. No vectors or tensors — training anything beyond a toy neuron means writing a lot of individual `Value`s, which doesn't scale.
- **No graph freeing.** Nothing ever drops old graph nodes, so long-running training loops build up memory. Fine for a demo, not for real training.
- **Single-threaded.** `Rc`/`RefCell` are not thread-safe (that's `Arc`/`Mutex`'s job) — this can't be parallelized without a rewrite.

## Natural next step

Replace the scalar `f64` in `ValueData` with a `Vec<f64>` plus a shape (a minimal tensor), and vectorize each `Op`'s forward/backward logic. Same graph-and-chain-rule mechanism throughout — only the math per node changes from a single number to an array operation.
