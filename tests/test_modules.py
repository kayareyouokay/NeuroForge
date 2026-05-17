import numpy as np

from neuroforge import Tensor
from neuroforge.nn import Conv2D, LayerNorm, Linear, Sequential, TransformerEncoderBlock
from neuroforge.nn.losses import mse_loss
from neuroforge.optim import Adam


def test_linear_model_can_fit_tiny_regression() -> None:
    np.random.seed(1)
    model = Sequential(Linear(2, 8), LayerNorm(8), Linear(8, 1))
    opt = Adam(model.parameters(), lr=0.03)
    x = Tensor(np.random.randn(16, 2).astype(np.float32))
    y = Tensor((2.0 * x.data[:, :1] - 0.5 * x.data[:, 1:2]).astype(np.float32))

    first = None
    last = None
    for _ in range(40):
        pred = model(x)
        loss = mse_loss(pred, y)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = loss.item()

    assert last is not None and first is not None
    assert last < first


def test_conv2d_forward_and_backward_shapes() -> None:
    conv = Conv2D(3, 4, kernel_size=3, padding=1)
    x = Tensor(np.random.randn(2, 3, 8, 8).astype(np.float32), requires_grad=True)
    y = conv(x)
    assert y.shape == (2, 4, 8, 8)
    y.mean().backward()
    assert conv.weight.grad is not None
    assert conv.weight.grad.shape == conv.weight.shape
    assert x.grad is not None


def test_transformer_encoder_preserves_shape() -> None:
    block = TransformerEncoderBlock(embed_dim=16, num_heads=4, dropout=0.0)
    x = Tensor((0.1 * np.random.randn(2, 5, 16)).astype(np.float32), requires_grad=True)
    y = block(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert any(p.grad is not None for p in block.parameters())
