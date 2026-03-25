"""
CCPGP Verification Suite — 12 tests covering all core properties.

Run: python -m pytest tests/ -v
  or: python tests/test_all.py
"""

import math
import random
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ccpgp import (
    CCANeuron, CCPGPSynapse, CCPGPNetwork, PROVENANCE_GATES,
    PyramidalCCA, FastPVCCA, SlowSSTCCA, VIPCCA, TRNCCA,
    HeteroSynapse, HeteroNetwork,
)

PASS = 0
FAIL = 0


def check(name: str, condition: bool) -> None:
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    if not condition:
        FAIL += 1
    else:
        PASS += 1
    print(f"  [{status}] {name}")


def test_01_dual_timescale():
    """Fast membrane + slow narrative operate at different scales."""
    print("\nT01: Dual Timescale Dynamics")
    n = CCANeuron("t1", tau_fast=5.0, tau_slow=2000.0, theta_base=0.3, gamma=0.15)
    for t in range(500):
        n.step(0.15 if t % 12 < 5 else 0.0)
    check("neuron spiked", n.spike_count > 5)
    check("narrative accumulated", n.n > 0.001)
    check("threshold shifted up", n.theta > n.theta_base + 0.0001)
    print(f"    spikes={n.spike_count}, n={n.n:.4f}, θ shift={n.theta - n.theta_base:.4f}")


def test_02_provenance_gating():
    """Different provenance tags produce different signal strengths."""
    print("\nT02: Provenance Gating")
    spikes = {}
    for prov in PROVENANCE_GATES:
        post = CCANeuron(f"p_{prov}", tau_fast=5.0, tau_slow=5000.0, theta_base=0.3)
        syn = CCPGPSynapse("x", f"p_{prov}", w=0.8, prov=prov)
        total = 0
        for t in range(300):
            I = syn.transmit(1.0 if t % 8 == 0 else 0.0)
            total += int(post.step(I))
        spikes[prov] = total
    check("confirmed > speculation", spikes["confirmed_fact"] > spikes["speculation"])
    print(f"    spikes: {spikes}")


def test_03_constitutional_constraint():
    """Hard constraint blocks forbidden weight updates."""
    print("\nT03: Constitutional Constraint")
    net = CCPGPNetwork("t3")
    net.add_n(CCANeuron("s", tau_fast=5.0, theta_base=0.25))
    net.add_n(CCANeuron("a", tau_fast=5.0, theta_base=0.3))
    s_spec = CCPGPSynapse("s", "a", w=0.35, prov="speculation")
    s_fact = CCPGPSynapse("s", "a", w=0.35, prov="confirmed_fact")
    net.add_s(s_spec)
    net.add_s(s_fact)
    net.add_c(lambda s, w: not (s.prov == "speculation" and w > 0.4))
    net.regret = 0.5
    for t in range(2000):
        net.step({"s": 0.15 if t % 6 < 3 else 0.0})
    check("speculation capped at 0.4", s_spec.w <= 0.401)
    check("some updates blocked", s_spec.blocked > 0)
    check("fact synapse grew larger", s_fact.w > s_spec.w)
    print(f"    spec w={s_spec.w:.4f} blocked={s_spec.blocked}, fact w={s_fact.w:.4f}")


def test_04_regret_modulation():
    """Positive regret strengthens, negative weakens learning."""
    print("\nT04: Regret Modulation")
    ws = {}
    for rv in [-0.8, 0.0, 0.8]:
        n1 = CCANeuron("a", tau_fast=5.0, theta_base=0.25)
        n2 = CCANeuron("b", tau_fast=5.0, theta_base=0.3)
        sy = CCPGPSynapse("a", "b", w=0.3, prov="confirmed_fact")
        allow = lambda s, w: 0.0 <= w <= 2.0
        for t in range(600):
            n1.step(0.15 if t % 8 < 3 else 0.0)
            ps = 1.0 if t % 8 == 0 else 0.0
            n2.step(sy.transmit(ps) + 0.04)
            sy.update(ps, n2.last_spike, allow, regret=rv)
        ws[rv] = sy.w
    check("w(+0.8) > w(0) > w(-0.8)", ws[0.8] > ws[0.0] > ws[-0.8])
    print(f"    w(-0.8)={ws[-0.8]:.4f}, w(0)={ws[0.0]:.4f}, w(+0.8)={ws[0.8]:.4f}")


def test_05_identity_divergence():
    """Two identical systems diverge under different experiences."""
    print("\nT05: Identity Divergence")

    def mk():
        net = CCPGPNetwork("id")
        net.add_n(CCANeuron("sA", tau_fast=5.0, tau_slow=800.0, theta_base=0.25, gamma=0.2))
        net.add_n(CCANeuron("sB", tau_fast=5.0, tau_slow=800.0, theta_base=0.25, gamma=0.2))
        net.add_n(CCANeuron("integ", tau_fast=6.0, tau_slow=800.0, theta_base=0.20, gamma=0.2))
        net.add_n(CCANeuron("out", tau_fast=6.0, tau_slow=800.0, theta_base=0.20, gamma=0.2))
        net.add_s(CCPGPSynapse("sA", "integ", w=0.5, prov="confirmed_fact"))
        net.add_s(CCPGPSynapse("sB", "integ", w=0.5, prov="confirmed_fact"))
        net.add_s(CCPGPSynapse("integ", "out", w=0.4, prov="behavioral_inference"))
        net.add_c(lambda s, w: 0.0 <= w <= 2.0)
        return net

    h1, h2 = mk(), mk()
    random.seed(42)
    for day in range(100):
        for t in range(100):
            e1 = {}
            if random.random() < 0.7:
                e1["sA"] = 0.15 + random.gauss(0, 0.02)
            if random.random() < 0.15:
                e1["sB"] = 0.10
            h1.regret = random.gauss(0.15, 0.3)
            h1.step(e1)
            e2 = {}
            if random.random() < 0.15:
                e2["sA"] = 0.10
            if random.random() < 0.7:
                e2["sB"] = 0.15 + random.gauss(0, 0.02)
            h2.regret = random.gauss(-0.1, 0.3)
            h2.step(e2)

    w1 = [s.w for s in h1.synapses]
    w2 = [s.w for s in h2.synapses]
    w_div = sum(abs(a - b) for a, b in zip(w1, w2))
    exp_ok = h1.synapses[0].w > h2.synapses[0].w and h2.synapses[1].w > h1.synapses[1].w
    check("systems diverged (w_div > 0.01)", w_div > 0.01)
    check("experience shaped correct pathways", exp_ok)
    print(f"    w_divergence={w_div:.4f}")


def test_06_stability():
    """10000 steps with random input, no NaN/Inf."""
    print("\nT06: Long-run Stability")

    def mk():
        net = CCPGPNetwork("stable")
        net.add_n(CCANeuron("sA", tau_fast=5.0, tau_slow=800.0, theta_base=0.25, gamma=0.2))
        net.add_n(CCANeuron("sB", tau_fast=5.0, tau_slow=800.0, theta_base=0.25, gamma=0.2))
        net.add_n(CCANeuron("integ", tau_fast=6.0, tau_slow=800.0, theta_base=0.20, gamma=0.2))
        net.add_n(CCANeuron("out", tau_fast=6.0, tau_slow=800.0, theta_base=0.20, gamma=0.2))
        net.add_s(CCPGPSynapse("sA", "integ", w=0.5, prov="confirmed_fact"))
        net.add_s(CCPGPSynapse("sB", "integ", w=0.5, prov="confirmed_fact"))
        net.add_s(CCPGPSynapse("integ", "out", w=0.4, prov="behavioral_inference"))
        net.add_c(lambda s, w: 0.0 <= w <= 2.0)
        return net

    ns = mk()
    random.seed(99)
    bad = False
    for t in range(10000):
        e = {}
        if random.random() < 0.5:
            e["sA"] = 0.12 + random.gauss(0, 0.02)
        if random.random() < 0.3:
            e["sB"] = 0.10
        ns.regret = random.gauss(0.0, 0.4)
        ns.step(e)
        for s in ns.synapses:
            if math.isnan(s.w) or math.isinf(s.w):
                bad = True
        for n in ns.neurons.values():
            if math.isnan(n.v) or math.isnan(n.n):
                bad = True
    check("no NaN/Inf in 10000 steps", not bad)
    check("all weights in bounds", all(0.0 <= s.w <= 2.0 for s in ns.synapses))


def test_07_learning_speed():
    """Higher provenance = faster learning."""
    print("\nT07: Provenance Learning Speed")
    speeds = {}
    for prov in ["confirmed_fact", "behavioral_inference", "speculation"]:
        n1 = CCANeuron("a", tau_fast=5.0, theta_base=0.25)
        n2 = CCANeuron("b", tau_fast=5.0, theta_base=0.25)
        sy = CCPGPSynapse("a", "b", w=0.3, prov=prov)
        allow = lambda s, w: 0.0 <= w <= 2.0
        for t in range(1000):
            n1.step(0.15 if t % 8 < 3 else 0.0)
            ps = 1.0 if t % 8 == 0 else 0.0
            n2.step(sy.transmit(ps) + 0.05)
            sy.update(ps, n2.last_spike, allow, regret=0.2)
        speeds[prov] = abs(sy.w - 0.3)
    check("confirmed >= behavioral >= speculation",
          speeds["confirmed_fact"] >= speeds["behavioral_inference"] >= speeds["speculation"])
    print(f"    |Δw|: {', '.join(f'{k}={v:.4f}' for k, v in speeds.items())}")


def test_08_cabm_scenarios():
    """CABM scenario classification with lateral inhibition."""
    print("\nT08: CABM Scenario Classification")
    net = CCPGPNetwork("cabm")
    for nid in ["danger", "request", "observation", "authority", "emotion", "routine"]:
        net.add_n(CCANeuron(f"in_{nid}", tau_fast=5.0, theta_base=0.2))
    for nid in ["reflex", "watchful", "deliberative", "ambient"]:
        net.add_n(CCANeuron(f"out_{nid}", tau_fast=5.0, theta_base=0.3))
    conns = [
        ("in_danger", "out_reflex", 0.7, "confirmed_fact"),
        ("in_danger", "out_watchful", 0.2, "confirmed_fact"),
        ("in_request", "out_deliberative", 0.6, "social_normative"),
        ("in_observation", "out_watchful", 0.55, "behavioral_inference"),
        ("in_authority", "out_deliberative", 0.4, "social_normative"),
        ("in_emotion", "out_watchful", 0.35, "affective_inference"),
        ("in_routine", "out_ambient", 0.65, "confirmed_fact"),
    ]
    for pre, post, w, prov in conns:
        net.add_s(CCPGPSynapse(pre, post, w=w, prov=prov))
    outputs = ["out_reflex", "out_watchful", "out_deliberative", "out_ambient"]
    for a in outputs:
        for b in outputs:
            if a != b:
                net.add_s(CCPGPSynapse(a, b, w=0.25, prov="confirmed_fact", inhibitory=True))
    net.add_c(lambda s, w: 0.0 <= w <= 2.0)

    scenarios = [
        ({"in_danger": 0.25, "in_authority": 0.1}, "out_reflex"),
        ({"in_request": 0.2, "in_routine": 0.05}, "out_deliberative"),
        ({"in_routine": 0.22, "in_request": 0.04}, "out_ambient"),
        ({"in_emotion": 0.14, "in_observation": 0.1}, "out_watchful"),
    ]
    random.seed(42)
    correct = 0
    for pattern, expected in scenarios:
        winner, _ = net.classify(pattern, ticks=60)
        if winner == expected:
            correct += 1
    check(f"scenario accuracy >= 75% ({correct}/{len(scenarios)})", correct >= 3)


def test_09_manifold_proof():
    """20000 adversarial steps, 0 constraint violations."""
    print("\nT09: Constitutional Manifold Proof")
    net = CCPGPNetwork("proof")
    net.add_n(CCANeuron("a", tau_fast=3.0, theta_base=0.15))
    net.add_n(CCANeuron("b", tau_fast=3.0, theta_base=0.15))
    syn = CCPGPSynapse("a", "b", w=0.55, prov="confirmed_fact", Ap=0.05, Am=0.001)
    net.add_s(syn)
    net.add_c(lambda s, w: w <= 0.6)
    violations = 0
    random.seed(123)
    for _ in range(20000):
        net.regret = 2.0
        net.step({"a": 0.3, "b": 0.05})
        if syn.w > 0.601:
            violations += 1
    check("zero violations in 20000 adversarial steps", violations == 0)
    print(f"    max w={syn.w:.6f}, blocked={syn.blocked}")


def test_10_sleep_consolidation():
    """Narrative survives night, Day 2 > Day 1."""
    print("\nT10: Sleep Consolidation")
    n = CCANeuron("sleep", tau_fast=5.0, tau_slow=500.0, theta_base=0.25, gamma=0.2)
    for t in range(500):
        n.step(0.15 if t % 10 < 5 else 0.0)
    n_day1 = n.n
    for t in range(500):
        n.step(0.0)
    n_night = n.n
    for t in range(500):
        n.step(0.15 if t % 10 < 5 else 0.0)
    n_day2 = n.n
    check("narrative survived night (n > 0.001)", n_night > 0.001)
    check("night < day1 (decayed)", n_night < n_day1)
    check("day2 > day1 (accumulated on residue)", n_day2 > n_day1)
    print(f"    day1={n_day1:.4f}, night={n_night:.4f}, day2={n_day2:.4f}")


def test_11_provenance_escalation():
    """Upgrading provenance from speculation to confirmed accelerates learning."""
    print("\nT11: Provenance Escalation")
    n1 = CCANeuron("x", tau_fast=5.0, theta_base=0.2)
    n2 = CCANeuron("y", tau_fast=5.0, theta_base=0.25)
    syn = CCPGPSynapse("x", "y", w=0.3, prov="speculation")
    allow = lambda s, w: 0.0 <= w <= 2.0
    for t in range(400):
        n1.step(0.12 if t % 8 < 3 else 0.0)
        ps = 1.0 if t % 8 == 0 else 0.0
        n2.step(syn.transmit(ps) + 0.04)
        syn.update(ps, n2.last_spike, allow, regret=0.2)
    w_spec = syn.w
    syn.prov = "confirmed_fact"
    for t in range(400):
        n1.step(0.12 if t % 8 < 3 else 0.0)
        ps = 1.0 if t % 8 == 0 else 0.0
        n2.step(syn.transmit(ps) + 0.04)
        syn.update(ps, n2.last_spike, allow, regret=0.2)
    w_conf = syn.w
    check("learning accelerated after upgrade", w_conf > w_spec * 1.5)
    print(f"    w_spec={w_spec:.4f}, w_confirmed={w_conf:.4f}")


def test_12_multi_constraint():
    """Multiple constraints satisfied simultaneously."""
    print("\nT12: Multi-Constraint")
    net = CCPGPNetwork("multi")
    net.add_n(CCANeuron("s1", tau_fast=5.0, theta_base=0.2))
    net.add_n(CCANeuron("s2", tau_fast=5.0, theta_base=0.2))
    net.add_n(CCANeuron("act", tau_fast=5.0, theta_base=0.2))
    syn1 = CCPGPSynapse("s1", "act", w=0.3, prov="behavioral_inference")
    syn2 = CCPGPSynapse("s2", "act", w=0.3, prov="affective_inference")
    net.add_s(syn1)
    net.add_s(syn2)
    net.add_c(lambda s, w: not (s.prov == "affective_inference" and w > 0.5))

    def sum_check(s, wc):
        total = wc if s is syn1 or s is syn2 else 0.0
        for x in [syn1, syn2]:
            if x is not s:
                total += x.w
        return total <= 1.2
    net.add_c(sum_check)
    net.add_c(lambda s, w: 0.0 <= w <= 2.0)
    random.seed(77)
    for t in range(3000):
        net.regret = random.gauss(0.3, 0.2)
        net.step({"s1": 0.15 if t % 6 < 3 else 0.0, "s2": 0.12 if t % 7 < 3 else 0.0})
    check("affective <= 0.5", syn2.w <= 0.501)
    check("sum <= 1.2", syn1.w + syn2.w <= 1.201)
    check("updates were blocked", syn1.blocked > 0)
    print(f"    w1={syn1.w:.4f}, w2={syn2.w:.4f}, sum={syn1.w + syn2.w:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("CCPGP Verification Suite — 12 Tests")
    print("=" * 60)
    test_01_dual_timescale()
    test_02_provenance_gating()
    test_03_constitutional_constraint()
    test_04_regret_modulation()
    test_05_identity_divergence()
    test_06_stability()
    test_07_learning_speed()
    test_08_cabm_scenarios()
    test_09_manifold_proof()
    test_10_sleep_consolidation()
    test_11_provenance_escalation()
    test_12_multi_constraint()
    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
    print("=" * 60)
    if FAIL > 0:
        sys.exit(1)
