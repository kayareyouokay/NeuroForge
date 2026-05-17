import numpy as np

from neuroforge.nn import Linear, Sequential
from neuroforge.optim import SGD, StepLR
from neuroforge.training.checkpoint import load_checkpoint, save_checkpoint


def test_scheduler_updates_learning_rate() -> None:
    model = Sequential(Linear(2, 2))
    opt = SGD(model.parameters(), lr=0.1)
    sched = StepLR(opt, step_size=2, gamma=0.5)
    assert sched.step() == 0.1
    assert sched.step() == 0.05


def test_checkpoint_roundtrip(tmp_path) -> None:
    model = Sequential(Linear(2, 3))
    path = tmp_path / "model.npz"
    state = {k: v.copy() for k, v in model.state_dict().items()}
    save_checkpoint(path, model, epoch=3)
    for p in model.parameters():
        p.data += 1.0
    metadata = load_checkpoint(path, model)
    assert metadata["epoch"] == 3
    for name, value in model.state_dict().items():
        np.testing.assert_allclose(value, state[name])
