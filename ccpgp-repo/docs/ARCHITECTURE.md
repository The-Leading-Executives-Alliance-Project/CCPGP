# CCPGP Architecture

## Overview

CCPGP (Constitutionally Constrained Provenance-Gated Plasticity) is a synaptic
learning rule for spiking neural networks. It extends standard Spike-Timing-Dependent
Plasticity (STDP) with three additional mechanisms that operate at the synapse level.

This document describes the mathematical formulation, neuron model, synapse model,
and network-level behaviour in enough detail to reimplement CCPGP from scratch.

---

## 1. Neuron Model — CCA (Constitutionally Constrained Adaptive)

Each neuron maintains two state variables at different timescales.

### Fast membrane potential (millisecond scale)

```
dv/dt = -(v - v_rest) / τ_fast + I_total
```

This is standard leaky integrate-and-fire (LIF) dynamics. `I_total` is the sum
of all provenance-gated synaptic inputs plus any external current.

### Slow narrative potential (day scale)

```
dn/dt = -(n - n_baseline) / τ_slow + β · spike
```

`τ_slow` is typically 100–1000× larger than `τ_fast`. The narrative potential
accumulates the neuron's activity history over long timescales. It decays slowly
and never resets to zero unless the neuron is completely inactive for an extended
period.

### Threshold modulation

```
θ(t) = θ_base + γ · n(t)
```

The firing threshold is not constant. It rises as the narrative potential
accumulates. A frequently active neuron gradually becomes harder to activate.
This implements a form of homeostatic adaptation driven by the neuron's own
experience history.

### Key property

Two neurons with identical parameters (τ_fast, θ_base, γ) will develop
different effective thresholds after different experience histories, because
their narrative potentials diverge. This is the mechanism for
**experience-driven individuality**.

### What this is not

The narrative potential is related to but distinct from Spike-Frequency
Adaptation (SFA), which operates on millisecond-to-second timescales and
serves a frequency regulation purpose. The narrative potential operates on
day-to-week timescales and serves an identity formation purpose. The concept
of a slow accumulator is not itself novel — what is novel is using it to
produce individualization in ethically constrained systems.

---

## 2. Synapse Model — CCPGP Update Rule

The weight update at each timestep is:

```
Δw = Δw_STDP × C(w, F) × P(p) × R(ρ)
```

### Δw_STDP — Spike-Timing-Dependent Plasticity

Standard additive STDP with eligibility traces:

```
If post-synaptic neuron fires: Δw += A+ · trace_pre
If pre-synaptic neuron fires:  Δw -= A- · trace_post
```

Traces decay exponentially with time constant `τ_trace`.

### C(w, F) — Constitutional Gate

A binary function:
- Returns 1 if the candidate weight `w + Δw` lies within the permissible
  weight manifold `W_permissible`.
- Returns 0 otherwise. The update is silently discarded.

This is a **hard projection**, not a soft penalty. Regardless of the magnitude
of the STDP signal, the regret signal, or the provenance gate value, the
constitutional check cannot be bypassed. Mathematically:

```
W_permissible = { w ∈ ℝⁿ | g_k(w) ≤ 0, ∀ k ∈ {1, ..., K} }
```

Each constraint `g_k` encodes a specific axiom. Examples:
- Speculation-tagged synapses cannot exceed weight 0.4
- The sum of inference-tagged synapses to an action neuron cannot exceed 1.2
- Reflex pathway weights can only increase, never decrease

In our verification suite, 20,000 adversarial steps with maximum regret signal
produced **zero** constraint violations.

### P(p) — Provenance Gate

Each synapse carries a provenance tag `p` from a fixed vocabulary:

| Provenance | Gate α(p) | Meaning |
|------------|-----------|---------|
| confirmed_fact | 1.00 | Directly observed and verified |
| social_normative | 0.85 | Established relationship rules |
| behavioral_inference | 0.50 | Inferred from observed behavior |
| affective_inference | 0.30 | Inferred emotional state |
| speculation | 0.10 | Internal narrative hypothesis |

The gate value scales both signal transmission **and** learning rate.
A speculation-tagged synapse with weight 1.0 transmits only 0.1 units of
current. It also learns at 10% of the rate of a confirmed_fact synapse.

### R(ρ) — Regret Modulation

```
R(ρ) = 1 + λ · ρ
```

where `ρ` is a global counterfactual regret signal. When the system's chosen
action outperforms alternatives (`ρ > 0`), learning is amplified. When it
underperforms (`ρ < 0`), learning is dampened.

This differs from reward-modulated STDP (R-STDP) in a specific way: R-STDP
uses an absolute reward signal ("was the outcome good?"). CCPGP uses a
counterfactual comparison ("was the outcome better than what would have
happened with a different choice?"). These are mathematically different
quantities.

### What is novel, what is not

- **STDP itself**: well-established since Bi & Poo (1998). Not novel.
- **Provenance-gated learning rate**: we found no prior art for scaling
  synaptic learning rate by a semantic source-type tag. Novel.
- **Constitutional hard projection on ethical axioms in SNN plasticity**:
  hard weight constraints exist in deep learning (e.g., NeurIPS 2025), but
  applying them to encode ethical axioms in synaptic plasticity is novel.
- **Counterfactual regret-modulated STDP**: R-STDP uses reward; CCPGP uses
  counterfactual regret. Different mathematical quantity. Novel.
- **The four-mechanism combination**: no prior work combines all four. Novel.

---

## 3. Heterogeneous Substrate

The homogeneous model uses identical CCA neurons throughout. The heterogeneous
model introduces five neuron types with different parameters, inspired by
major cortical neuron classes:

| Type | Biological analogue | τ_fast | θ_base | γ | Role |
|------|-------------------|--------|--------|---|------|
| PyramidalCCA | Glutamatergic pyramidal | 10 | 0.42 | 0.12 | Main excitatory carrier |
| FastPVCCA | PV+ basket/chandelier | 4 | 0.30 | 0.05 | Fast inhibition, competition |
| SlowSSTCCA | SST+ Martinotti | 18 | 0.38 | 0.18 | Slow context suppression |
| VIPCCA | VIP+ disinhibitory | 12 | 0.36 | 0.10 | Selective pathway release |
| TRNCCA | Thalamic reticular | 8 | 0.46 | 0.08 | Sensory relay gating |

The HeteroSynapse supports three plasticity modes:
- `ccpgp`: Full CCPGP rule (excitatory pathways)
- `fixed`: No plasticity (structural connections)
- `inhibitory_homeostatic`: Mirror balancing rule (inhibitory connections)

### Individualization benchmark

In a controlled test (Alpha 19 of the CABM project), two identical
heterogeneous networks were trained on different experience distributions
for 100 simulated days. When presented with the same ambiguous input probe:

| Method | Individualization Score | Interpretation |
|--------|----------------------|----------------|
| Static classifier | 0.000 | No divergence — identical response |
| Homogeneous CCPGP | 0.201 | Slight divergence |
| Heterogeneous CCPGP | **2.000** | Complete divergence — opposite responses |

The heterogeneous network produced 10× more individualization than the
homogeneous version, and infinite improvement over the static baseline.
This is a direct consequence of different neuron types accumulating narrative
potential at different rates and in different directions.

---

## 4. Limitations and Honest Assessment

**Classification accuracy.** On small labeled datasets (tens of examples),
a standard linear classifier outperforms CCPGP in raw classification accuracy.
CCPGP is not designed for static classification — it is designed for
on-device continuous adaptation and individualization.

**Scale.** All verification was performed with networks of 5–50 neurons. We
have not tested CCPGP at scales of thousands or millions of neurons. Scaling
behaviour is an open question.

**Formal convergence proof.** We have empirical evidence of stability
(10,000+ steps without NaN/Inf, bounded weights) but no formal proof of
convergence to a fixed point within the constitutional manifold under
stationary input distributions. This is planned future work.

**Biological plausibility.** The provenance gate has no known direct biological
correlate. The constitutional constraint has no biological correlate. These
are engineering mechanisms inspired by neuroscience but designed for safety,
not biological realism.

**The narrative potential timescale.** We chose day-scale accumulation based
on engineering requirements (household android individualization over weeks).
The optimal timescale for other applications is unknown.

---

## 5. Relationship to CABM

CCPGP was developed as the neural substrate for CABM (Cooperative Android
Base Model), a cognitive architecture for household androids. In that context:

- CCPGP manages local attention shaping, interrupt salience, and event
  formation — tasks where on-device adaptation is valuable.
- CABM's constitutional core (deterministic rule engine) retains final
  authority over action legality — tasks where safety guarantees are required.
- The two systems are complementary: CCPGP learns what is useful; the
  constitutional core prevents what is forbidden.

CCPGP can be used independently of CABM for any application that requires
adaptive spiking networks with source-awareness and safety constraints.
