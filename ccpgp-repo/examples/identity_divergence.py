"""
Identity Divergence: Two identical systems become different individuals
through different experiences. This is the core demonstration of CCPGP's
individualization capability.
"""

import random
from ccpgp import CCANeuron, CCPGPSynapse, CCPGPNetwork

random.seed(42)


def build_network(name: str) -> CCPGPNetwork:
    net = CCPGPNetwork(name)
    net.add_n(CCANeuron("sA", tau_fast=5.0, tau_slow=800.0, theta_base=0.25, gamma=0.2))
    net.add_n(CCANeuron("sB", tau_fast=5.0, tau_slow=800.0, theta_base=0.25, gamma=0.2))
    net.add_n(CCANeuron("integrator", tau_fast=6.0, tau_slow=800.0, theta_base=0.20, gamma=0.2))
    net.add_n(CCANeuron("output", tau_fast=6.0, tau_slow=800.0, theta_base=0.20, gamma=0.2))
    net.add_s(CCPGPSynapse("sA", "integrator", w=0.5, prov="confirmed_fact"))
    net.add_s(CCPGPSynapse("sB", "integrator", w=0.5, prov="confirmed_fact"))
    net.add_s(CCPGPSynapse("integrator", "output", w=0.4, prov="behavioral_inference"))
    net.add_c(lambda s, w: 0.0 <= w <= 2.0)
    return net


# Two identical starting points
home_a = build_network("Family_A")
home_b = build_network("Family_B")

# Family A: mostly verbal interaction (sensor A dominant)
# Family B: mostly physical/spatial events (sensor B dominant)
print("Simulating 100 days of different experiences...\n")
for day in range(100):
    for tick in range(100):
        # Family A: sensor_A active 70%, sensor_B 15%
        ext_a = {}
        if random.random() < 0.70:
            ext_a["sA"] = 0.15 + random.gauss(0, 0.02)
        if random.random() < 0.15:
            ext_a["sB"] = 0.10
        home_a.regret = random.gauss(0.15, 0.3)
        home_a.step(ext_a)

        # Family B: sensor_A 15%, sensor_B active 70%
        ext_b = {}
        if random.random() < 0.15:
            ext_b["sA"] = 0.10
        if random.random() < 0.70:
            ext_b["sB"] = 0.15 + random.gauss(0, 0.02)
        home_b.regret = random.gauss(-0.1, 0.3)
        home_b.step(ext_b)

# Compare
print("After 100 days (10,000 ticks per system):")
print("=" * 55)
print(f"{'Metric':<25} {'Family A':>14} {'Family B':>14}")
print("-" * 55)

for i, label in enumerate(["sA→integ", "sB→integ", "integ→out"]):
    wa = home_a.synapses[i].w
    wb = home_b.synapses[i].w
    print(f"{label + ' weight':<25} {wa:>14.4f} {wb:>14.4f}")

print()
for nid in ["sA", "sB", "integrator", "output"]:
    na = home_a.neurons[nid].n
    nb = home_b.neurons[nid].n
    print(f"{nid + ' narrative':<25} {na:>14.4f} {nb:>14.4f}")

print()
for nid in ["sA", "sB"]:
    ta = home_a.neurons[nid].theta
    tb = home_b.neurons[nid].theta
    print(f"{nid + ' threshold':<25} {ta:>14.4f} {tb:>14.4f}")

w_div = sum(abs(home_a.synapses[i].w - home_b.synapses[i].w) for i in range(3))
print(f"\nTotal weight divergence: {w_div:.4f}")
print(f"Experience correctly shaped weights: "
      f"{home_a.synapses[0].w > home_b.synapses[0].w and home_b.synapses[1].w > home_a.synapses[1].w}")
