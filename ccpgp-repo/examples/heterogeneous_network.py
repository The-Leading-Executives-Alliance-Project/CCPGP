"""
Heterogeneous Network: Five neuron types working together
to classify attention modes with lateral inhibition.

Demonstrates:
  - Pyramidal excitation carrying signal
  - PV+ fast inhibition enforcing winner-take-all
  - SST+ slow modulation providing context
  - VIP+ disinhibition selectively opening pathways
  - TRN gating controlling sensory relay
"""

import random
from ccpgp import (
    PyramidalCCA, FastPVCCA, SlowSSTCCA, VIPCCA, TRNCCA,
    HeteroSynapse, HeteroNetwork,
)

random.seed(42)

# Build network
net = HeteroNetwork("attention_classifier")

# Input layer: TRN gates for sensory channels
net.add_n(TRNCCA("in_danger"))
net.add_n(TRNCCA("in_request"))
net.add_n(TRNCCA("in_routine"))

# Processing: pyramidal + modulators
net.add_n(PyramidalCCA("pyr_reflex"))
net.add_n(PyramidalCCA("pyr_delib"))
net.add_n(PyramidalCCA("pyr_ambient"))
net.add_n(FastPVCCA("pv_comp"))       # lateral inhibition
net.add_n(SlowSSTCCA("sst_context"))  # slow context
net.add_n(VIPCCA("vip_gate"))         # disinhibition

# Feedforward excitatory connections (CCPGP plasticity)
net.add_s(HeteroSynapse("in_danger", "pyr_reflex", w=0.6, prov="confirmed_fact"))
net.add_s(HeteroSynapse("in_request", "pyr_delib", w=0.5, prov="social_normative"))
net.add_s(HeteroSynapse("in_routine", "pyr_ambient", w=0.55, prov="confirmed_fact"))

# Pyramidal → PV (fixed excitatory drive to inhibitor)
for pyr in ["pyr_reflex", "pyr_delib", "pyr_ambient"]:
    net.add_s(HeteroSynapse(pyr, "pv_comp", w=0.3, plasticity="fixed"))

# PV → Pyramidal (lateral inhibition, homeostatic plasticity)
for pyr in ["pyr_reflex", "pyr_delib", "pyr_ambient"]:
    net.add_s(HeteroSynapse("pv_comp", pyr, w=0.25, inhibitory=True, plasticity="inhibitory_homeostatic"))

# SST slow context modulation
net.add_s(HeteroSynapse("sst_context", "pyr_delib", w=0.15, inhibitory=True, plasticity="fixed"))
net.add_s(HeteroSynapse("pyr_delib", "sst_context", w=0.2, plasticity="fixed"))

# VIP disinhibits SST (inhibits the inhibitor = releases the pathway)
net.add_s(HeteroSynapse("vip_gate", "sst_context", w=0.3, inhibitory=True, plasticity="fixed"))

# Constitutional constraint
net.add_c(lambda s, w: 0.0 <= w <= 2.0)
net.add_c(lambda s, w: not (s.prov == "speculation" and w > 0.35))

# Run three scenarios
scenarios = [
    ("Danger detected",  {"in_danger": 0.25}, "pyr_reflex"),
    ("Help requested",   {"in_request": 0.20}, "pyr_delib"),
    ("Routine activity", {"in_routine": 0.22}, "pyr_ambient"),
]

for name, pattern, expected_winner in scenarios:
    # Reset fast variables between scenarios
    net.reset_fast()
    spike_counts = {nid: 0 for nid in net.neurons if nid.startswith("pyr_")}

    for t in range(60):
        ext = {k: v + random.gauss(0, 0.01) for k, v in pattern.items()}
        # Activate VIP gate during deliberative scenarios
        if "in_request" in pattern:
            ext["vip_gate"] = 0.08
        spikes = net.step(ext)
        for nid in spike_counts:
            spike_counts[nid] += int(spikes.get(nid, 0))

    winner = max(spike_counts, key=spike_counts.get)
    correct = "✓" if winner == expected_winner else "✗"
    print(f"{correct} {name:20s} → {winner:15s} (spikes: {spike_counts})")

# Show network state
print("\nNeuron states after all scenarios:")
for nid, info in net.summary()["neurons"].items():
    print(f"  {nid:15s} [{info['type']:22s}] θ={info['theta']:.3f}  n={info['n']:.4f}  spikes={info['spikes']}")
