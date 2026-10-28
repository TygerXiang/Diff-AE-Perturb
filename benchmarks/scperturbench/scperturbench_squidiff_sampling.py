"""Sampling-policy helpers for Squidiff's DDIM loop.

Why this exists
---------------
Squidiff keeps guided-diffusion's ``clip_denoised`` flag but rewrote the ``else`` branch of
``process_xstart`` inside ``GaussianDiffusion.p_mean_variance`` (``Squidiff/diffusion.py``)::

    def process_xstart(x):
        if denoised_fn is not None:
            x = denoised_fn(x)
        if clip_denoised:
            return x.clamp(-1, 1)
        return x.clamp(0,)          # <- Squidiff only; guided-diffusion returns `x`

``sample_squidiff.sampler.pred`` calls ``ddim_sample_loop`` with its defaults, so
``clip_denoised=False`` and every ``pred_xstart`` is clamped to be non-negative.  At high
noise levels ``pred_xstart`` is ill-conditioned -- ``1 / sqrt(alpha_cumprod)`` reaches ~190
at t=999 -- so it is a zero-mean quantity with a huge scale, and clamping it at zero
rectifies that noise into a large one-sided drift that accumulates over the 1000-step chain.

``POLICIES`` therefore exposes the three ``process_xstart`` behaviours:

``upstream``  exactly Squidiff as published, i.e. ``x.clamp(0,)``  (uses ``sampler.pred``)
``neg1_1``    Squidiff's own ``clip_denoised=True`` branch, ``x.clamp(-1, 1)``
``none``      guided-diffusion's original behaviour, i.e. no clamp at all

The DDIM loop itself is always Squidiff's own ``ddim_sample_loop``; only ``process_xstart``
differs, and it differs through an instance-level monkeypatch of ``p_mean_variance``.
"""

from __future__ import annotations

import copy
import types

import numpy as np
import torch as th

POLICIES = ("upstream", "none", "neg1_1")


def patched_p_mean_variance(policy: str):
    """Faithful copy of Squidiff's ``p_mean_variance`` with a different ``process_xstart``."""
    if policy not in POLICIES:
        raise ValueError(f"unknown sampling policy {policy!r}; expected one of {POLICIES}")
    from Squidiff.diffusion import ModelVarType, _extract_into_tensor

    def p_mean_variance(self, model, x, t, clip_denoised=False, denoised_fn=None, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        _, c = x.shape[:2]
        assert t.shape == (x.shape[0],)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in (ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE):
            model_output, _ = th.split(model_output, c, dim=1)
        model_variance = np.append(self.posterior_variance[1], self.betas[1:])
        model_log_variance = np.log(model_variance)
        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        pred_xstart_raw = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)

        def process_xstart(value):
            if denoised_fn is not None:
                value = denoised_fn(value)
            if policy == "none":
                return value
            if policy == "neg1_1":
                return value.clamp(-1, 1)
            return value.clamp(0,)

        pred_xstart = process_xstart(pred_xstart_raw)
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "pred_xstart_raw": pred_xstart_raw,
        }

    return p_mean_variance


def sample_with_policy(sampler, z_sem, gene_size: int, policy: str):
    """Draw cells with Squidiff's DDIM loop under one ``process_xstart`` policy."""
    if policy == "upstream":
        return sampler.pred(z_sem, gene_size=gene_size)
    if policy == "neg1_1":
        return sampler.diffusion.ddim_sample_loop(
            sampler.model,
            shape=(z_sem.shape[0], gene_size),
            model_kwargs={"z_mod": z_sem},
            noise=None,
            clip_denoised=True,
        )
    local = copy.copy(sampler.diffusion)
    local.p_mean_variance = types.MethodType(patched_p_mean_variance("none"), local)
    return local.ddim_sample_loop(
        sampler.model,
        shape=(z_sem.shape[0], gene_size),
        model_kwargs={"z_mod": z_sem},
        noise=None,
    )


def sample_progressive(sampler, z_sem, gene_size: int, policy: str):
    """Generator form used for per-step tracing; mirrors :func:`sample_with_policy`."""
    local = copy.copy(sampler.diffusion)
    local.p_mean_variance = types.MethodType(patched_p_mean_variance(policy), local)
    return local, local.ddim_sample_loop_progressive(
        sampler.model,
        shape=(z_sem.shape[0], gene_size),
        model_kwargs={"z_mod": z_sem},
        device=z_sem.device,
    )
