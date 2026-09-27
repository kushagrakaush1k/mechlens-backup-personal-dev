import torch

from transformer_lens.ActivationCache import ActivationCache


def test_repr_includes_keys_without_representing_values():
    class ReprMustNotBeCalled(torch.Tensor):
        def __repr__(self):
            raise AssertionError("ActivationCache.__repr__ must not represent cached values")

    cache_dict: dict[str, torch.Tensor] = {
        "hook_embed": torch.zeros(1).as_subclass(ReprMustNotBeCalled),
        "blocks.0.hook_resid_pre": torch.zeros(1).as_subclass(ReprMustNotBeCalled),
    }
    cache = ActivationCache(cache_dict, model=None)

    representation = repr(cache)

    for key in cache_dict:
        assert key in representation
