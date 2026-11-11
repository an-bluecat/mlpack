"""Teaching implementations of normalized ELU+1 linear attention.

Run from the repository root:
    .venv/bin/python courses/linear_attention_course/lab.py

Tensor convention: (batch, heads, time, features). These are transparent CPU
references, not optimized GPU kernels. No changes to the TorchCode exercises.
"""

import json
import torch
import torch.nn.functional as F


def phi(x):
    """ELU(alpha=1)+1; positive mathematically, may round to zero in float."""
    return F.elu(x) + 1


def dense_reference(q, k, v, causal=False, eps=1e-6):
    """Explicit T x T matrix for verifying our kernel, NOT softmax attention."""
    qf, kf = phi(q), phi(k)
    weights = qf @ kf.transpose(-1, -2)
    if causal:
        mask = torch.ones(weights.shape[-2:], device=q.device, dtype=torch.bool).tril()
        weights = weights.masked_fill(~mask, 0)
    return (weights @ v) / weights.sum(-1, keepdim=True).clamp_min(eps)


def linear_attention(q, k, v, eps=1e-6):
    """Bidirectional: all query positions can read all key/value positions."""
    qf, kf = phi(q), phi(k)
    state = kf.transpose(-1, -2) @ v        # (B, H, M, Dv)
    key_sum = kf.sum(dim=-2)               # (B, H, M)
    numerator = qf @ state                # (B, H, T, Dv)
    denominator = (qf * key_sum.unsqueeze(-2)).sum(-1, keepdim=True)
    return numerator / denominator.clamp_min(eps)


def recurrent_attention(q, k, v, initial_state=None, eps=1e-6):
    """Inclusive causal attention; returns output and state for continuation.

    Uses out-of-place additions so autograd can retain the earlier states.
    For inference, call inside torch.no_grad(). For full BPTT, do not detach
    the returned state between chunks. At least one token is required.
    """
    qf, kf = phi(q), phi(k)
    if initial_state is None:
        state = q.new_zeros(*q.shape[:2], qf.shape[-1], v.shape[-1])
        key_sum = q.new_zeros(*q.shape[:2], qf.shape[-1])
    else:
        state, key_sum = initial_state
    outputs = []
    for t in range(q.shape[-2]):
        qt, kt, vt = qf[:, :, t], kf[:, :, t], v[:, :, t]
        state = state + kt.unsqueeze(-1) * vt.unsqueeze(-2)
        key_sum = key_sum + kt
        numerator = (qt.unsqueeze(-2) @ state).squeeze(-2)
        denominator = (qt * key_sum).sum(-1, keepdim=True)
        outputs.append(numerator / denominator.clamp_min(eps))
    return torch.stack(outputs, dim=-2), (state, key_sum)


def chunked_attention(q, k, v, chunk_size=4, eps=1e-6):
    """Causal attention: incoming state plus a dense triangle inside each chunk.

    With chunk size C fixed, local scores and mixing cost O(T*C*(M+Dv)),
    and state work costs O(T*M*Dv). This is not a production parallel scan.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    qf, kf = phi(q), phi(k)
    state = q.new_zeros(*q.shape[:2], qf.shape[-1], v.shape[-1])
    key_sum = q.new_zeros(*q.shape[:2], qf.shape[-1])
    outputs = []
    for start in range(0, q.shape[-2], chunk_size):
        qc = qf[:, :, start:start + chunk_size]
        kc = kf[:, :, start:start + chunk_size]
        vc = v[:, :, start:start + chunk_size]
        local = qc @ kc.transpose(-1, -2)
        mask = torch.ones(local.shape[-2:], device=q.device, dtype=torch.bool).tril()
        local = local.masked_fill(~mask, 0)
        numerator = qc @ state + local @ vc
        denominator = (qc * key_sum.unsqueeze(-2)).sum(-1, keepdim=True)
        denominator = denominator + local.sum(-1, keepdim=True)
        outputs.append(numerator / denominator.clamp_min(eps))
        state = state + kc.transpose(-1, -2) @ vc
        key_sum = key_sum + kc.sum(-2)
    return torch.cat(outputs, dim=-2)


def example():
    # Raw Q/K chosen so phi is exactly [[1,1],[1,2],[2,1]].
    q = torch.tensor([[0., 0.], [0., 1.], [1., 0.]], dtype=torch.float64)
    k = q.clone()
    v = torch.tensor([[2., 0.], [0., 4.], [6., 2.]], dtype=torch.float64)
    return q[None, None], k[None, None], v[None, None]


def verify():
    """Independent dense reference, gradients, causal isolation and streaming."""
    torch.manual_seed(7)
    cases = 0
    gradient_errors = []
    for b, h, t, dk, dv in [(1, 1, 1, 2, 3), (2, 3, 7, 4, 5), (1, 2, 9, 3, 2)]:
        xs = [torch.randn(*shape, dtype=torch.float64, requires_grad=True)
              for shape in [(b, h, t, dk), (b, h, t, dk), (b, h, t, dv)]]
        probe = torch.randn(b, h, t, dv, dtype=torch.float64)
        for causal in (False, True):
            expected = dense_reference(*xs, causal=causal)
            candidates = ([linear_attention(*xs)] if not causal else
                          [recurrent_attention(*xs)[0]] +
                          [chunked_attention(*xs, chunk_size=c) for c in (1, 3, t + 1)])
            ref_grads = torch.autograd.grad((expected * probe).sum(), xs, retain_graph=True)
            for actual in candidates:
                torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
                grads = torch.autograd.grad((actual * probe).sum(), xs, retain_graph=True)
                for got, want in zip(grads, ref_grads):
                    torch.testing.assert_close(got, want, rtol=1e-9, atol=1e-10)
                    gradient_errors.append((got - want).abs().max().item())
                cases += 1
        full, _ = recurrent_attention(*xs)
        split = max(1, t // 2)
        first, state = recurrent_attention(*(x[:, :, :split] for x in xs))
        if split < t:
            second, _ = recurrent_attention(*(x[:, :, split:] for x in xs), initial_state=state)
            torch.testing.assert_close(torch.cat([first, second], -2), full)
            # Full gradients must survive the boundary too.
            stream_grads = torch.autograd.grad(
                (torch.cat([first, second], -2) * probe).sum(), xs, retain_graph=True)
            full_grads = torch.autograd.grad((full * probe).sum(), xs, retain_graph=True)
            for got, want in zip(stream_grads, full_grads):
                torch.testing.assert_close(got, want, rtol=1e-9, atol=1e-10)
        changed = [x.detach().clone() for x in xs]
        changed[1][:, :, split:] += 20
        changed[2][:, :, split:] -= 100
        altered, _ = recurrent_attention(*changed)
        torch.testing.assert_close(altered[:, :, :split], full[:, :, :split])

    tiny = tuple((torch.randn(1, 1, 3, d, dtype=torch.float64) * .1 + .4)
                 .requires_grad_() for d in (2, 2, 3))
    assert torch.autograd.gradcheck(lambda *x: recurrent_attention(*x)[0], tiny)
    # Negative-tail rounding produces zeros; the clamp must keep output finite.
    q = torch.full((1, 1, 2, 2), -100., requires_grad=True)
    out = linear_attention(q, q, torch.ones_like(q))
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(q.grad).all()

    q, k, v = example()
    qf, kf = phi(q), phi(k)
    state, key_sum = kf.transpose(-1, -2) @ v, kf.sum(-2)
    causal, _ = recurrent_attention(q, k, v)
    softmax = torch.softmax(q @ k.transpose(-1, -2) / q.shape[-1] ** .5, -1) @ v
    # Last-token, first-output scalar loss: (y - 1)^2.
    qg, kg, vg = [x.clone().requires_grad_() for x in (q, k, v)]
    y, _ = recurrent_attention(qg, kg, vg)
    loss = (y[0, 0, -1, 0] - 1).square()
    loss.backward()
    return {
        "torch": torch.__version__, "seed": 7,
        "forward_gradient_cases": cases,
        "max_gradient_error": max(gradient_errors),
        "feature_vectors": qf[0, 0].tolist(),
        "weights": (qf @ kf.transpose(-1, -2))[0, 0].tolist(),
        "state": state[0, 0].tolist(), "key_sum": key_sum[0, 0].tolist(),
        "bidirectional": linear_attention(q, k, v)[0, 0].tolist(),
        "causal": causal[0, 0].tolist(),
        "softmax": softmax[0, 0].tolist(),
        "loss": loss.item(), "value_gradient": vg.grad[0, 0].tolist(),
        "query_gradient": qg.grad[0, 0].tolist(),
        "key_gradient": kg.grad[0, 0].tolist(),
    }


if __name__ == "__main__":
    results = verify()
    print(json.dumps(results, indent=2))
    print("PASS: forward/gradient agreement, finite differences, streaming and causality.")
