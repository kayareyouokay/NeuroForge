import numpy as np

from neuroforge import Tensor
from neuroforge.nn.losses import cross_entropy


def test_scalar_graph_accumulates_gradients() -> None:
    x = Tensor([1.0, -2.0, 3.0], requires_grad=True)
    y = ((x * x) + 2.0 * x).sum()
    y.backward()
    np.testing.assert_allclose(x.grad, np.array([4.0, -2.0, 8.0], dtype=np.float32))


def test_broadcast_backward_reduces_to_operand_shape() -> None:
    x = Tensor(np.ones((4, 3), dtype=np.float32), requires_grad=True)
    b = Tensor(np.zeros((3,), dtype=np.float32), requires_grad=True)
    loss = (x + b).sum()
    loss.backward()
    np.testing.assert_allclose(b.grad, np.full((3,), 4.0, dtype=np.float32))


def test_matmul_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(7)
    x_data = rng.normal(size=(2, 3)).astype(np.float32)
    w_data = rng.normal(size=(3, 4)).astype(np.float32)
    x = Tensor(x_data, requires_grad=True)
    w = Tensor(w_data, requires_grad=True)
    loss = (x @ w).sum()
    loss.backward()

    eps = 1e-3
    numerical = np.zeros_like(w_data)
    for i in range(w_data.shape[0]):
        for j in range(w_data.shape[1]):
            plus = w_data.copy()
            minus = w_data.copy()
            plus[i, j] += eps
            minus[i, j] -= eps
            numerical[i, j] = ((x_data @ plus).sum() - (x_data @ minus).sum()) / (2 * eps)
    np.testing.assert_allclose(w.grad, numerical, rtol=1e-2, atol=1e-2)


def test_cross_entropy_backpropagates_to_logits() -> None:
    logits = Tensor([[1.0, 0.0, -1.0], [0.0, 2.0, -0.5]], requires_grad=True)
    loss = cross_entropy(logits, np.array([0, 1]))
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape
