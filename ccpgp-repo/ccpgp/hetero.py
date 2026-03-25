"""
CCPGP Heterogeneous Substrate — Multiple neuron roles for richer dynamics.

Five biologically-inspired computational roles:
  - PyramidalCCA      : Main excitatory carrier
  - FastPVCCA         : Fast basket/chandelier-like inhibition
  - SlowSSTCCA        : Slow contextual modulation
  - VIPCCA            : Disinhibitory gating
  - TRNCCA            : Thalamic reticular gate

Copyright (c) 2026 LEAP A.I Industries Ltd.
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple

from .core import CCANeuron, PROVENANCE_GATES

__all__ = [
    "PyramidalCCA",
    "FastPVCCA",
    "SlowSSTCCA",
    "VIPCCA",
    "TRNCCA",
    "HeteroSynapse",
    "HeteroNetwork",
]


class PyramidalCCA(CCANeuron):
    """Main excitatory carrier — cortical pyramidal analogue."""
    def __init__(self, nid: str) -> None:
        super().__init__(nid, tau_fast=10.0, tau_slow=3000.0, theta_base=0.42, gamma=0.12)
        self.type_name = "pyramidal"


class FastPVCCA(CCANeuron):
    """Fast basket / chandelier analogue — rapid competition & output suppression."""
    def __init__(self, nid: str) -> None:
        super().__init__(nid, tau_fast=4.0, tau_slow=1800.0, theta_base=0.30, gamma=0.05)
        self.type_name = "pv_fast_inhibitory"

    def step(self, I_total: float, dt: float = 1.0) -> float:
        self.v += dt * (-self.v / self.tau_fast + 1.15 * I_total)
        self.theta = self.theta_base + self.gamma * self.n
        spike = 1.0 if self.v >= self.theta else 0.0
        if spike:
            self.v = -0.05
            self.spike_count += 1
        self.n += dt * (-self.n / self.tau_slow + 0.010 * spike)
        self.last_spike = spike
        return spike


class SlowSSTCCA(CCANeuron):
    """Slow contextual suppressor — SST+ Martinotti analogue."""
    def __init__(self, nid: str) -> None:
        super().__init__(nid, tau_fast=18.0, tau_slow=4500.0, theta_base=0.38, gamma=0.18)
        self.type_name = "sst_slow_modulator"


class VIPCCA(CCANeuron):
    """Disinhibitory gate — VIP+ analogue that selectively releases pathways."""
    def __init__(self, nid: str) -> None:
        super().__init__(nid, tau_fast=12.0, tau_slow=3200.0, theta_base=0.36, gamma=0.10)
        self.type_name = "vip_disinhibitory"


class TRNCCA(CCANeuron):
    """Thalamic reticular gate — sparse, cautious sensory relay controller."""
    def __init__(self, nid: str) -> None:
        super().__init__(nid, tau_fast=8.0, tau_slow=2500.0, theta_base=0.46, gamma=0.08)
        self.type_name = "trn_gate"


@dataclass
class HeteroSynapse:
    """Synapse supporting multiple plasticity modes."""
    pre: str
    post: str
    w: float = 0.3
    prov: str = "confirmed_fact"
    inhibitory: bool = False
    plasticity: str = "ccpgp"  # ccpgp | fixed | inhibitory_homeostatic
    Ap: float = 0.012
    Am: float = 0.006
    tau_tr: float = 20.0
    tr_pre: float = field(default=0.0, repr=False)
    tr_post: float = field(default=0.0, repr=False)
    blocked: int = field(default=0, repr=False)
    updates: int = field(default=0, repr=False)

    def gate(self) -> float:
        return PROVENANCE_GATES.get(self.prov, 0.1)

    def transmit(self, spike: float) -> float:
        sign = -1.0 if self.inhibitory else 1.0
        return sign * self.w * spike * self.gate()

    def update(self, pre_s: float, post_s: float, check_fn: Callable, regret: float = 0.0, dt: float = 1.0) -> None:
        if self.plasticity == "fixed":
            return
        self.tr_pre *= 1.0 - dt / self.tau_tr
        self.tr_pre += pre_s
        self.tr_post *= 1.0 - dt / self.tau_tr
        self.tr_post += post_s

        dw = 0.0
        if self.plasticity == "ccpgp":
            if post_s > 0.5:
                dw += self.Ap * self.tr_pre
            if pre_s > 0.5:
                dw -= self.Am * self.tr_post
            w_candidate = self.w + dw * self.gate() * (1.0 + 0.5 * regret)
        elif self.plasticity == "inhibitory_homeostatic":
            if post_s > 0.5:
                dw -= 0.004 * self.tr_pre
            if pre_s > 0.5:
                dw += 0.002 * self.tr_post
            w_candidate = self.w + dw
        else:
            return

        if abs(dw) < 1e-12:
            return
        self.updates += 1
        if check_fn(self, w_candidate):
            self.w = max(0.0, min(2.0, w_candidate))
        else:
            self.blocked += 1


class HeteroNetwork:
    """Network of heterogeneous CCA neurons with mixed synapse types."""

    def __init__(self, name: str = "hetero") -> None:
        self.name = name
        self.neurons: Dict[str, CCANeuron] = {}
        self.synapses: List[HeteroSynapse] = []
        self.constraints: List[Callable] = []
        self.regret: float = 0.0

    def add_n(self, n: CCANeuron) -> None:
        self.neurons[n.id] = n

    def add_s(self, s: HeteroSynapse) -> None:
        self.synapses.append(s)

    def add_c(self, fn: Callable) -> None:
        self.constraints.append(fn)

    def check(self, syn: HeteroSynapse, w: float) -> bool:
        return all(fn(syn, w) for fn in self.constraints)

    def step(self, ext: Dict[str, float], dt: float = 1.0) -> Dict[str, float]:
        currents: Dict[str, float] = defaultdict(float)
        for s in self.synapses:
            if self.neurons[s.pre].last_spike > 0.5:
                currents[s.post] += s.transmit(1.0)
        for nid, val in ext.items():
            currents[nid] += val
        spikes: Dict[str, float] = {}
        for nid, n in self.neurons.items():
            spikes[nid] = n.step(currents.get(nid, 0.0), dt)
        for s in self.synapses:
            s.update(spikes.get(s.pre, 0.0), spikes.get(s.post, 0.0), self.check, self.regret, dt)
        return spikes

    def reset_fast(self) -> None:
        for n in self.neurons.values():
            n.reset_fast()
        for s in self.synapses:
            s.tr_pre = 0.0
            s.tr_post = 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "neurons": {nid: {"type": n.type_name, "v": round(n.v, 4), "n": round(n.n, 4),
                               "theta": round(n.theta, 4), "spikes": n.spike_count}
                        for nid, n in self.neurons.items()},
            "synapses": [{"pre": s.pre, "post": s.post, "w": round(s.w, 4), "prov": s.prov,
                          "plasticity": s.plasticity, "inhibitory": s.inhibitory,
                          "blocked": s.blocked, "updates": s.updates}
                         for s in self.synapses],
        }
