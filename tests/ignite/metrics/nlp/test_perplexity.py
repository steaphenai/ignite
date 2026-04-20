import pytest
import torch
import torch.nn.functional as F

from ignite.exceptions import NotComputableError
from ignite.metrics.nlp import Perplexity


def test_zero_sample():
    ppl = Perplexity()
    ppl.reset()
    with pytest.raises(NotComputableError):
        ppl.compute()


def test_compute_matches_manual():
    torch.manual_seed(42)
    ppl = Perplexity()
    ppl.reset()

    y_pred = torch.randn(4, 10, 5)
    y = torch.randint(0, 10, (4, 5))

    ppl.update((y_pred, y))

    nll_manual = F.cross_entropy(y_pred, y, reduction="sum").item()
    ppl_manual = torch.exp(torch.tensor(nll_manual / y.numel())).item()

    assert abs(ppl.compute() - ppl_manual) < 1e-4


def test_token_weighted_accumulation():
    """Token-weighted accumulation must differ from naive batch average."""
    torch.manual_seed(0)
    ppl = Perplexity()
    ppl.reset()

    # Two batches with different sequence lengths
    b1_pred = torch.randn(2, 5, 4)
    b1_y = torch.randint(0, 5, (2, 4))
    b2_pred = torch.randn(3, 5, 10)
    b2_y = torch.randint(0, 5, (3, 10))

    ppl.update((b1_pred, b1_y))
    ppl.update((b2_pred, b2_y))

    nll1 = F.cross_entropy(b1_pred, b1_y, reduction="sum").item()
    nll2 = F.cross_entropy(b2_pred, b2_y, reduction="sum").item()
    total_tokens = b1_y.numel() + b2_y.numel()
    ppl_ref = torch.exp(torch.tensor((nll1 + nll2) / total_tokens)).item()

    assert abs(ppl.compute() - ppl_ref) < 1e-4


def test_returns_float():
    torch.manual_seed(1)
    ppl = Perplexity()
    ppl.reset()

    y_pred = torch.randn(2, 5, 3)
    y = torch.randint(0, 5, (2, 3))
    ppl.update((y_pred, y))

    result = ppl.compute()
    assert isinstance(result, float)


def test_invalid_y_pred_shape():
    ppl = Perplexity()
    ppl.reset()

    with pytest.raises(ValueError, match="y_pred must be at least 2-dimensional"):
        ppl.update((torch.tensor([1.0, 2.0]), torch.tensor([0])))


def test_reset_clears_state():
    torch.manual_seed(2)
    ppl = Perplexity()

    y_pred = torch.randn(2, 5, 3)
    y = torch.randint(0, 5, (2, 3))
    ppl.update((y_pred, y))

    ppl.reset()
    with pytest.raises(NotComputableError):
        ppl.compute()


def test_single_token():
    ppl = Perplexity()
    ppl.reset()

    y_pred = torch.randn(1, 5, 1)
    y = torch.randint(0, 5, (1, 1))
    ppl.update((y_pred, y))

    result = ppl.compute()
    assert result > 0
    assert isinstance(result, float)
