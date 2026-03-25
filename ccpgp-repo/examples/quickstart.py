"""
Quickstart: Build a 2-neuron network with provenance gating
and constitutional constraints in under 20 lines.
"""

from ccpgp import CCANeuron, CCPGPSynapse, CCPGPNetwork

# Build network
net = CCPGPNetwork("quickstart")
net.add_n(CCANeuron("sensor", tau_fast=5.0, theta_base=0.25))
net.add_n(CCANeuron("action", tau_fast=5.0, theta_base=0.30))

# Confirmed fact synapse (full learning rate)
net.add_s(CCPGPSynapse("sensor", "action", w=0.3, prov="confirmed_fact"))

# Constitutional constraint: speculation can never exceed 0.4
net.add_c(lambda s, w: not (s.prov == "speculation" and w > 0.4))
net.add_c(lambda s, w: 0.0 <= w <= 2.0)

# Run 200 steps with positive regret
net.regret = 0.3
for t in range(200):
    spikes = net.step({"sensor": 0.15 if t % 8 < 4 else 0.0})

# Print results
print(net.summary())
print(f"\nWeight grew from 0.3 to {net.synapses[0].w:.4f}")
print(f"Narrative potential: {net.neurons['action'].n:.4f}")
print(f"Threshold shift: {net.neurons['action'].theta - 0.30:.4f}")
