<h1 align="center">CCPGP</h1>
<h3 align="center">Constitutionally Constrained Provenance-Gated Plasticity</h3>

<p align="center">
  <em>A novel synaptic learning rule for ethically bounded adaptive systems</em>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-brightgreen.svg" alt="Python">
  <img src="https://img.shields.io/badge/dependencies-zero-orange.svg" alt="Dependencies">
  <img src="https://img.shields.io/badge/tests-22%2F22-success.svg" alt="Tests">
</p>

---

## What is CCPGP?

CCPGP is a **learning rule for spiking neural networks** that embeds three mechanisms directly into synaptic plasticity:

| Mechanism | What it does | Why it matters |
|-----------|-------------|----------------|
| **Provenance Gating** | Scales learning rate by source reliability | A confirmed observation teaches 10× faster than a speculation |
| **Constitutional Constraints** | Hard-projects weights onto a permissible manifold | Certain weight configurations are *structurally impossible*, not just penalized |
| **Counterfactual Regret** | Modulates plasticity by outcome comparison | "Was this the best choice?" strengthens good pathways, weakens bad ones |

Together, these produce a system that **becomes uniquely shaped by experience** while remaining **provably incapable of learning unsafe patterns**.

---

## The Core Equation

```
Δw = Δw_STDP  ×  C(w, F)  ×  P(p)  ×  R(ρ)
      ─────      ───────      ────      ────
      Spike      Constitu-    Prove-    Regret
      Timing     tional       nance     Modu-
      Dependent  Gate         Gate      lation
      Plasticity (binary)     (0→1)     (scalar)
```

**Δw_STDP** — Standard spike-timing rule: if post fires after pre, strengthen; if before, weaken.

**C(w, F)** — Constitutional check: returns 1 if the new weight is in the permissible set, 0 if blocked. **This is a hard wall, not a soft penalty.** No gradient, no matter how strong, can push through it.

**P(p)** — Provenance gate: `α(p)` scales by source type. Values:

```
confirmed_fact ········· 1.00  ████████████████████
social_normative ······· 0.85  █████████████████
behavioral_inference ··· 0.50  ██████████
affective_inference ···· 0.30  ██████
speculation ············ 0.10  ██
```

**R(ρ)** — Regret modulation: `R = 1 + λ·ρ`. Positive regret (chose well) amplifies learning. Negative regret (chose poorly) dampens it.

---

## Dual-Timescale Neuron

Each CCA neuron has two state variables:

```
Fast (ms)                          Slow (days)
─────────                          ───────────
Membrane potential v               Narrative potential n
  dv/dt = -v/τ_fast + I             dn/dt = -n/τ_slow + β·spike

Resets after spike                 Accumulates over lifetime
Same across individuals            Unique to each individual

         ┌──────────────────────────────────────────┐
         │  Threshold:  θ(t) = θ_base + γ · n(t)    │
         │                                          │
         │  ► Experience raises the threshold       │
         │  ► Same neuron, different history =       │
         │    different sensitivity                  │
         └──────────────────────────────────────────┘
```

---

## Identity Divergence

Two **identical** systems, exposed to different experiences for 100 simulated days, become **different individuals**:

```
                    Family A              Family B
                  (high stimulation      (low stimulation
                   on sensor A)           on sensor A)
                  ─────────────          ─────────────
Day 0             w_A = 0.500            w_A = 0.500        ← identical
                  n_A = 0.000            n_A = 0.000

Day 100           w_A = 1.997            w_A = 1.419        ← diverged
                  n_A = 1.173            n_A = 0.048

                  ┌──────────────────────────────────┐
                  │  Weight divergence:    0.904       │
                  │  Narrative divergence: 2.363       │
                  │  Threshold divergence: 0.470       │
                  │                                    │
                  │  Same code + different experience   │
                  │  = different individual             │
                  └──────────────────────────────────┘
```

---

## Heterogeneous Substrate

Five neuron types for richer dynamics:

```
 ┌──────────────┐
 │  Pyramidal    │ ◄── Main excitatory carrier (τ_fast=10, θ=0.42)
 │  (Glutamate)  │     Long-range relay, task pressure
 └──────┬───────┘
        │
        ├───────────────────┐
        │                   │
 ┌──────▼───────┐   ┌──────▼───────┐
 │  PV+ Fast    │   │  SST+ Slow   │
 │  Inhibitory  │   │  Modulator   │
 │  (τ=4, θ=.30)│   │ (τ=18, θ=.38)│
 │  Winner-take-│   │  Context     │
 │  all, sharp  │   │  suppression │
 └──────────────┘   └──────────────┘
        │                   │
        │            ┌──────▼───────┐
        │            │  VIP+        │
        │            │  Disinhibit  │
        │            │ (τ=12, θ=.36)│
        │            │  "Open the   │
        │            │   gate"      │
        │            └──────────────┘
        │
 ┌──────▼───────┐
 │  TRN Gate    │ ◄── Sensory relay controller (τ=8, θ=0.46)
 │  "Which      │     Decides what gets through
 │   signals    │
 │   matter?"   │
 └──────────────┘
```

---

## Quickstart

```bash
pip install ccpgp        # or: git clone & pip install -e .
```

```python
from ccpgp import CCANeuron, CCPGPSynapse, CCPGPNetwork

# Build a network
net = CCPGPNetwork("demo")
net.add_n(CCANeuron("sensor", theta_base=0.25))
net.add_n(CCANeuron("action", theta_base=0.30))
net.add_s(CCPGPSynapse("sensor", "action", w=0.3, prov="confirmed_fact"))

# Add constitutional constraint
net.add_c(lambda s, w: not (s.prov == "speculation" and w > 0.4))

# Run with regret signal
net.regret = 0.3
for t in range(200):
    spikes = net.step({"sensor": 0.15 if t % 8 < 4 else 0.0})

print(f"Weight: 0.3 → {net.synapses[0].w:.3f}")
print(f"Narrative potential: {net.neurons['action'].n:.4f}")
```

---

## Verification Results

All 12 core properties verified. Zero external dependencies.

| # | Test | Key Result | Status |
|---|------|-----------|--------|
| 1 | Dual Timescale | 43 spikes, n=0.569, θ shift +0.085 | ✅ |
| 2 | Provenance Gating | confirmed=38 spikes, speculation=0 | ✅ |
| 3 | Constitutional Constraint | cap=0.4, actual=0.398, 102 blocked | ✅ |
| 4 | Regret Modulation | w(−0.8)=1.08 < w(0)=1.60 < w(+0.8)=2.00 | ✅ |
| 5 | Identity Divergence | w_div=0.90, correct direction | ✅ |
| 6 | Stability | 10,000 steps, no NaN/Inf | ✅ |
| 7 | Learning Speed | confirmed Δw=1.69 >> speculation Δw=0.04 | ✅ |
| 8 | Scenario Classification | 7/7 with lateral inhibition | ✅ |
| 9 | Manifold Proof | 20,000 adversarial steps, 0 violations | ✅ |
| 10 | Sleep Consolidation | narrative survives night, Day 2 > Day 1 | ✅ |
| 11 | Provenance Escalation | speculation→confirmed accelerates 440× | ✅ |
| 12 | Multi-Constraint | 2 simultaneous constraints satisfied | ✅ |

```bash
python tests/test_all.py
# RESULTS: 22 passed, 0 failed, 22 total
```

---

## How CCPGP Differs from Existing Methods

| | STDP | R-STDP | Backprop | **CCPGP** |
|---|:---:|:---:|:---:|:---:|
| Source-aware learning | ✗ | ✗ | ✗ | **✓** |
| Hard ethical constraints | ✗ | ✗ | ✗ (soft) | **✓** |
| Counterfactual regret | ✗ | reward only | loss only | **✓** |
| Dual-timescale identity | ✗ | ✗ | ✗ | **✓** |
| Fully auditable updates | partial | partial | ✗ | **✓** |
| Online / on-device | ✓ | ✓ | batch only | **✓** |
| Zero dependencies | ✓ | varies | PyTorch etc. | **✓** |

---

## Project Structure

```
ccpgp/
├── ccpgp/
│   ├── __init__.py          # Public API
│   ├── core.py              # CCANeuron, CCPGPSynapse, CCPGPNetwork
│   └── hetero.py            # 5 neuron types + HeteroSynapse + HeteroNetwork
├── tests/
│   └── test_all.py          # 12 verification tests (22 assertions)
├── examples/
│   ├── quickstart.py        # Minimal working example
│   └── identity_divergence.py  # Two systems become different individuals
├── docs/
│   └── ccpgp-banner.svg     # Banner graphic
├── LICENSE                  # Apache 2.0
├── README.md                # This file
└── pyproject.toml           # Package metadata
```

---

## Citation

If you use CCPGP in academic work, please cite:

```bibtex
@software{ccpgp2026,
  title     = {CCPGP: Constitutionally Constrained Provenance-Gated Plasticity},
  author    = {{LEAP A.I Industries Ltd.}},
  year      = {2026},
  url       = {https://github.com/The-Leading-Executives-Alliance-Project/CCPGP},
  license   = {Apache-2.0},
}
```

---

## License

Copyright (c) 2026 **LEAP A.I Industries Ltd.**

Licensed under the [Apache License, Version 2.0](LICENSE).

---

<p align="center">
  <sub>CCPGP was developed as the neural substrate for the <a href="#">CABM</a> (Cooperative Android Base Model) project.</sub>
</p>
