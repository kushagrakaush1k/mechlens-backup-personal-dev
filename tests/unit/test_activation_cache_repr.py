import torch
from IPython.core.formatters import DisplayFormatter

from transformer_lens.ActivationCache import ActivationCache


def test_repr_includes_keys_without_representing_values():
    repr_calls = 0

    class ReprMustNotBeCalled(torch.Tensor):
        def __repr__(self):
            nonlocal repr_calls
            repr_calls += 1
            raise AssertionError("ActivationCache.__repr__ must not represent cached values")

    cache_dict: dict[str, torch.Tensor] = {
        "hook_embed": torch.zeros(1).as_subclass(ReprMustNotBeCalled),
        "blocks.0.hook_resid_pre": torch.zeros(1).as_subclass(ReprMustNotBeCalled),
    }
    cache = ActivationCache(cache_dict, model=None)

    representation = repr(cache)

    for key in cache_dict:
        assert key in representation

    assert repr_calls == 0

    bundle, _ = DisplayFormatter().format(cache)

    # IPython can catch formatter exceptions, so check calls as well as output.
    assert repr_calls == 0
    assert "text/plain" in bundle
    for key in cache_dict:
        assert key in bundle["text/plain"]
