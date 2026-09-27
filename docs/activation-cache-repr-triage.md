# ActivationCache representation: issue #1126 triage

Status: source investigation complete; checkout benchmark and regression-test execution pending. No production behavior change proposed yet.

## What the issue reports

As summarized by the reporter in this discussion, printing a large `ActivationCache` appears to hang a notebook. The suggested explanation is that representing the cache performs expensive work on its cached tensors. According to the follow-up clarification in this discussion, the reporter proposed truncating the repr as the fix. This is the reporter's proposal, not a reproduced diagnosis or an agreed implementation change.

The original [issue #1126](https://github.com/TransformerLensOrg/TransformerLens/issues/1126) could not be retrieved during this investigation. This is a paraphrase of the discussion, not a verified quotation from the issue. The precise failing expression, revision, environment, tensor device, key count, and stack trace remain unverified. In particular, “large” could mean many keys, large tensor payloads, or both.

## What we actually verified in the repo

The source review used the supplied copies of `ActivationCache.py`, `hook_points.py`, the activation-cache benchmarks, compatibility tests, and `pyproject.toml`. A complete importable checkout was not available in the investigation environment.

### The cache wrapper formats keys only

In `transformer_lens/ActivationCache.py`, `ActivationCache.__repr__` returns:

```python
return f"ActivationCache with keys {list(self.cache_dict.keys())}"
```

The class defines no custom `__str__`, `_repr_pretty_`, `_repr_html_`, `_repr_mimebundle_`, or `_ipython_display_`. For the ordinary string-keyed dictionary used here, the method traverses and formats keys; it does not represent cached values or the stored model, transfer tensors, or compute activations.

| Expression | Representation path |
| --- | --- |
| `repr(cache)` | Calls the keys-only method above. |
| `str(cache)` or `print(cache)` | Falls back to `__repr__`; printing also writes the resulting string. |
| Bare `cache` or `display(cache)` in a notebook | Default plain-text formatting is expected to use the representation; external formatters and inspectors require separate verification. |
| `repr(cache.cache_dict)` | Represents dictionary values as well as keys. |
| `repr(cache[key])` | Represents the selected value. |

An earlier isolated execution of the uploaded method used values and a model whose `__repr__` raises. `repr`, `str`, and redirected `print` completed without calling those representations. This was a structural check of the extracted method, **not** an import of the real checkout, a notebook reproduction, or the benchmark below.

### Hook registration has a separate representation path

In `transformer_lens/hook_points.py`, `HookPoint.add_hook` names its wrapper during registration:

```python
# annotate the `full_hook` with the string representation of the `hook` function
if isinstance(hook, partial):
    # partial.__repr__() can be extremely slow if arguments contain large objects, which
    # is common when caching tensors.
    full_hook.__name__ = f"partial({hook.func.__repr__()},...)"
else:
    full_hook.__name__ = hook.__repr__()
```

The backward wrapper uses the same partial guard:

```python
if isinstance(hook, partial):
    _bwd_hook_wrapper.__name__ = f"partial({hook.func.__repr__()},...)"
else:
    _bwd_hook_wrapper.__name__ = hook.__repr__()
```

Ordinary `repr(partial(fn, cache=populated_dict))` can recursively represent the bound dictionary's values. An isolated sentinel check confirmed that distinction and that the uploaded guarded naming branch avoids those values. The current guard still represents `hook.func`; a custom callable can have an expensive representation.

The supplied files do not include caching-hook construction in `HookedRootModule` or the bridge. They therefore do not establish which object is bound, whether it is populated at registration, or whether this path caused the reported hang. `HookPoint.__repr__` itself reports only its name and hook counts.

### Coverage and execution limits

The supplied benchmark checks cache structure and activation agreement; the supplied compatibility tests do not measure representation latency. The new `tests/unit/test_activation_cache_repr.py` imports the real class and checks this contract:

```python
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
```

The repository's pytest configuration enables runtime type checking via `--jaxtyping-packages=transformer_lens,beartype.beartype`. The original plain-object sentinels violated the constructor's `Dict[str, torch.Tensor]` annotation. They have now been replaced with real tensor subclasses whose `__repr__` raises, preserving both checks while satisfying the tensor-value contract. The test imports `torch`; the excerpt above omits imports. Execution under the repository's normal pytest configuration remains pending.

The existing `tests/acceptance/test_activation_cache.py` exercises model-backed logit attribution, residual decomposition, head/neuron results, and projection behavior. No production code or acceptance tests have changed on this branch, so these additions are not expected to alter that behavior. This is a source-based assessment, not a passing acceptance-test result.

The probe and test passed syntax checks, and the probe's CLI help was checked. Its benchmark could not start because the workspace lacks a complete checkout; Torch, IPython, and TransformerLens were also unavailable. The real regression test has **not** been run. No production source was modified.

## Measurements

**Pending: the checkout benchmark has not been run. No timing results are claimed here.** Earlier isolated method experiments are not results for this benchmark.

Run in the checkout's dependency environment:

```bash
python debugging/activation_cache_repr_probe.py
python -m pytest tests/unit/test_activation_cache_repr.py
```

The probe imports the real class after putting the repository root first on `sys.path`. It checks module, module-spec, class, and method source paths against `transformer_lens/ActivationCache.py`, then compares the imported method's code with code compiled from that file. Compilation is for comparison only; extracted code is never executed as a substitute. It prints the source hash and method and checks for source/method changes after measurement. These checks establish correspondence to the local file, not to an independently authenticated upstream commit.

Defaults: CPU float32 tensors with distinct initialized storage; 20 warmup calls; 9 timed batches of 100 `repr()` calls each. Importing, allocation, warmups, and output are outside timing. Each batch duration is divided by its call count; the table records minimum, median, and maximum of those per-call batch averages. The probe also prints all batch samples. `timeit` disables cyclic garbage collection during timing by default. These measurements exclude notebook rendering and GPU behavior.

The numeric key counts and tensor sizes below are **planned inputs**, not observations. Sweep A reuses identical keys; sweep B uses fixed-width key names and fixed tensor size.

| Sweep | Keys | Elements per tensor | Payload MiB | Repr characters | Min µs/call | Median µs/call | Max µs/call |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| A: tensor size | 64 | 1 | PENDING | PENDING | PENDING | PENDING | PENDING |
| A: tensor size | 64 | 1,024 | PENDING | PENDING | PENDING | PENDING | PENDING |
| A: tensor size | 64 | 65,536 | PENDING | PENDING | PENDING | PENDING | PENDING |
| A: tensor size | 64 | 262,144 | PENDING | PENDING | PENDING | PENDING | PENDING |
| B: key count | 16 | 16 | PENDING | PENDING | PENDING | PENDING | PENDING |
| B: key count | 64 | 16 | PENDING | PENDING | PENDING | PENDING | PENDING |
| B: key count | 256 | 16 | PENDING | PENDING | PENDING | PENDING | PENDING |
| B: key count | 1,024 | 16 | PENDING | PENDING | PENDING | PENDING | PENDING |
| B: key count | 4,096 | 16 | PENDING | PENDING | PENDING | PENDING | PENDING |

Record alongside the results: commit and working-tree changes **PENDING**; source path/hash **PENDING**; Python/PyTorch versions **PENDING**; OS/CPU **PENDING**; raw probe output **PENDING**; pytest result **PENDING**. For the reported notebook reproduction, also record notebook/IPython versions, frontend, device, and exact expression.

## Diagnosis

The reviewed implementation does not support the explanation that ordinary `repr(cache)` expands cached tensors. Its direct work depends on key count and key-string length, not tensor payload size. We expect approximately stable timings in sweep A and increasing work/output in sweep B; these are predictions, not measured conclusions.

The key list is unbounded, so sufficiently many keys can still make string construction or notebook output costly. Representing the raw dictionary or an individual tensor is a different operation. Hook registration can also invoke representation before any explicit display, but this checkout already guards ordinary partial arguments. None of these candidate mechanisms has been established as the cause of #1126.

First run the verified-import probe and regression test. Then reproduce the exact reported expression, separating cache creation with output suppressed, `text = repr(cache)`, and notebook display. If the stall precedes display, capture a stack trace during hook registration and inspect callable type, bound object, and whether the bound cache is populated. Compare the reporter's loaded source with the reviewed guard before attributing historical behavior to this checkout.

## Proposed response

> Could you share a minimal reproducer, your TransformerLens version, and the exact print or display expression you ran in the notebook?

This is a draft response for review; it has not been posted to the issue.
