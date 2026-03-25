"""
CCPGP Core — Constitutionally Constrained Provenance-Gated Plasticity

A novel synaptic learning rule for ethically bounded adaptive systems.
Zero external dependencies.

Copyright (c) 2026 LEAP A.I Industries Ltd.
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = [
    "PROVENANCE_GATES",
    "CCANeuron",
    "CCPGPSynapse",
    "CCPGPNetwork",
]

PROVENANCE_GATES: Dict[str, float] = {
    "confirmed_fact": 1.0,
    "social_normative": 0.85,
    "behavioral_inference": 0.5,
    "affective_inference": 0.3,
    "speculation": 0.1,
}


class CCANeuron:
    """
    Constitutionally Constrained Adaptive Neuron.

    Dual-timescale dynamics:
      - Fast membrane potential v (millisecond scale): LIF dynamics
      - Slow narrative potential n (day scale): experience accumulator

    The narrative potential modulates the firing threshold:
        θ(t) = θ_base + γ · n(t)

    Two identical neurons diverge after different experience histories.

    Parameters
    ----------
    nid : str
        Unique neuron identifier.
    tau_fast : float
        Fast membrane time constant.
    tau_slow : float
        Slow narrative time constant (>> tau_fast).
    theta_base : float
        Base firing threshold.
    gamma : float
        Narrative-to-threshold coupling strength.
    """

    __slots__ = (
        "id", "v", "n", "theta_base", "theta", "tau_fast",
        "tau_slow", "gamma", "last_spike", "spike_count", "type_name",
    )

    def __init__(
        self,
        nid: str,
        tau_fast: float = 10.0,
        tau_slow: float = 3000.0,
        theta_base: float = 0.5,
        gamma: float = 0.15,
    ) -> None:
        self.id = nid
        self.v: float = 0.0
        self.n: float = 0.0
        self.theta_base = theta_base
        self.theta = theta_base
        self.tau_fast = tau_fast
        self.tau_slow = tau_slow
        self.gamma = gamma
        self.last_spike: float = 0.0
        self.spike_count: int = 0
        self.type_name: str = "cca"

    def step(self, I_total: float, dt: float = 1.0) -> float:
        """Advance one timestep. Returns 1.0 if neuron spiked, else 0.0."""
        self.v += dt * (-self.v / self.tau_fast + I_total)
        self.theta = self.theta_base + self.gamma * self.n
        spike = 1.0 if self.v >= self.theta else 0.0
        if spike:
            self.v = 0.0
            self.spike_count += 1
        self.n += dt * (-self.n / self.tau_slow + 0.015 * spike)
        self.last_spike = spike
        return spike

    def reset_fast(self) -> None:
        """Reset fast variables only. Preserves narrative potential."""
        self.v = 0.0
        self.last_spike = 0.0

    def reset_all(self) -> None:
        """Full reset including narrative (identity discontinuity)."""
        self.v = 0.0
        self.n = 0.0
        self.theta = self.theta_base
        self.last_spike = 0.0
        self.spike_count = 0


class CCPGPSynapse:
    """
    Constitutionally Constrained Provenance-Gated Synapse.

    Weight update:
        Δw = Δw_STDP × C(w, F) × P(p) × R(ρ)

    Where:
        Δw_STDP : Standard spike-timing-dependent plasticity
        C(w, F) : Constitutional gate — hard projection (binary)
        P(p)    : Provenance gate — α(p) learning rate scaling
        R(ρ)    : Regret modulation — 1 + λ·ρ

    Parameters
    ----------
    pre, post : str
        Pre- and post-synaptic neuron IDs.
    w : float
        Initial weight.
    prov : str
        Provenance tag (key in PROVENANCE_GATES).
    Ap, Am : float
        LTP / LTD amplitudes.
    tau_tr : float
        Eligibility trace time constant.
    inhibitory : bool
        If True, transmits negative current and skips plasticity.
    """

    __slots__ = (
        "pre", "post", "w", "prov", "Ap", "Am", "tau_tr",
        "tr_pre", "tr_post", "blocked", "updates", "inhibitory",
    )

    def __init__(
        self,
        pre: str,
        post: str,
        w: float = 0.3,
        prov: str = "confirmed_fact",
        Ap: float = 0.012,
        Am: float = 0.006,
        tau_tr: float = 20.0,
        inhibitory: bool = False,
    ) -> None:
        self.pre = pre
        self.post = post
        self.w = w
        self.prov = prov
        self.Ap = Ap
        self.Am = Am
        self.tau_tr = tau_tr
        self.tr_pre: float = 0.0
        self.tr_post: float = 0.0
        self.blocked: int = 0
        self.updates: int = 0
        self.inhibitory = inhibitory

    def gate(self) -> float:
        """Provenance gate value α(p)."""
        return PROVENANCE_GATES.get(self.prov, 0.1)

    def transmit(self, spike: float) -> float:
        """Compute provenance-gated synaptic current."""
        sign = -1.0 if self.inhibitory else 1.0
        return sign * self.w * spike * self.gate()

    def update(
        self,
        pre_s: float,
        post_s: float,
        check_fn: Callable[["CCPGPSynapse", float], bool],
        regret: float = 0.0,
        dt: float = 1.0,
    ) -> None:
        """Apply CCPGP learning rule with constitutional projection."""
        if self.inhibitory:
            return

        self.tr_pre *= 1.0 - dt / self.tau_tr
        self.tr_pre += pre_s
        self.tr_post *= 1.0 - dt / self.tau_tr
        self.tr_post += post_s

        dw = 0.0
        if post_s > 0.5:
            dw += self.Ap * self.tr_pre
        if pre_s > 0.5:
            dw -= self.Am * self.tr_post

        if abs(dw) < 1e-12:
            return

        P = self.gate()
        R = 1.0 + 0.5 * regret
        w_candidate = self.w + dw * P * R

        self.updates += 1
        if check_fn(self, w_candidate):
            self.w = max(0.0, min(2.0, w_candidate))
        else:
            self.blocked += 1

    def reset_traces(self) -> None:
        self.tr_pre = 0.0
        self.tr_post = 0.0


class CCPGPNetwork:
    """
    A network of CCA neurons connected by CCPGP synapses.

    Provides automatic current routing, constitutional constraint
    checking, global regret injection, and classification helpers.

    Example
    -------
    >>> net = CCPGPNetwork("demo")
    >>> net.add_n(CCANeuron("a", theta_base=0.25))
    >>> net.add_n(CCANeuron("b", theta_base=0.30))
    >>> net.add_s(CCPGPSynapse("a", "b", w=0.5))
    >>> net.add_c(lambda s, w: w <= 1.0)
    >>> net.regret = 0.3
    >>> spikes = net.step({"a": 0.15})
    """

    def __init__(self, name: str = "network") -> None:
        self.name = name
        self.neurons: Dict[str, CCANeuron] = {}
        self.synapses: List[CCPGPSynapse] = []
        self.constraints: List[Callable] = []
        self.regret: float = 0.0

    def add_n(self, neuron: CCANeuron) -> None:
        self.neurons[neuron.id] = neuron

    def add_s(self, synapse: CCPGPSynapse) -> None:
        self.synapses.append(synapse)

    def add_c(self, constraint_fn: Callable) -> None:
        """Add a constitutional constraint: fn(synapse, w_candidate) → bool."""
        self.constraints.append(constraint_fn)

    def check(self, synapse: CCPGPSynapse, w_candidate: float) -> bool:
        return all(fn(synapse, w_candidate) for fn in self.constraints)

    def step(self, external_inputs: Dict[str, float], dt: float = 1.0) -> Dict[str, float]:
        """Advance one timestep. Returns {neuron_id: spike}."""
        currents: Dict[str, float] = defaultdict(float)
        for syn in self.synapses:
            pre = self.neurons[syn.pre]
            if pre.last_spike > 0.5:
                currents[syn.post] += syn.transmit(1.0)
        for nid, I_ext in external_inputs.items():
            currents[nid] += I_ext

        spikes: Dict[str, float] = {}
        for nid, neuron in self.neurons.items():
            spikes[nid] = neuron.step(currents.get(nid, 0.0), dt)

        for syn in self.synapses:
            syn.update(
                spikes.get(syn.pre, 0.0),
                spikes.get(syn.post, 0.0),
                self.check,
                self.regret,
                dt,
            )
        return spikes

    def reset_fast(self) -> None:
        for n in self.neurons.values():
            n.reset_fast()
        for s in self.synapses:
            s.reset_traces()

    def classify(
        self,
        pattern: Dict[str, float],
        ticks: int = 50,
        output_prefix: str = "out_",
        noise: float = 0.005,
    ) -> Tuple[str, Dict[str, float]]:
        """Run a pattern and return (winner_id, spike_distribution)."""
        import random
        self.reset_fast()
        counts: Dict[str, int] = defaultdict(int)
        for _ in range(ticks):
            ext = {k: v + random.gauss(0, noise) if random.random() < 0.9 else 0.0
                   for k, v in pattern.items()}
            spikes = self.step(ext)
            for nid, s in spikes.items():
                if nid.startswith(output_prefix) and s > 0.5:
                    counts[nid] += 1
        if counts:
            total = sum(counts.values()) or 1
            dist = {k: v / total for k, v in counts.items()}
            return max(dist, key=dist.get), dist  # type: ignore
        return "none", {}

    def summary(self) -> Dict[str, Any]:
        return {
            "neurons": {
                nid: {"type": n.type_name, "v": round(n.v, 4), "n": round(n.n, 4),
                       "theta": round(n.theta, 4), "spikes": n.spike_count}
                for nid, n in self.neurons.items()
            },
            "synapses": [
                {"pre": s.pre, "post": s.post, "w": round(s.w, 4),
                 "prov": s.prov, "blocked": s.blocked, "updates": s.updates}
                for s in self.synapses if not s.inhibitory
            ],
        }
