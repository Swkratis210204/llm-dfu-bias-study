import numpy as np
from scipy.stats import skew, kurtosis

def get_sarle_bc(data):
    n = len(data)
    g = skew(data, bias=False)
    k = kurtosis(data, bias=False)
    return (g**2 + 1) / (k + 3 * ((n-1)**2 / ((n-2)*(n-3))))

def get_dfu(freqs):
    m = np.argmax(freqs)
    K = len(freqs)
    d = []
    for i in range(K):
        if m < i < K:
            d.append(freqs[i] - freqs[i-1])
        elif 0 < i < m:
            d.append(freqs[i] - freqs[i+1])
    return max(0, np.max(d)) if d else 0

def get_agreement_a(freqs):
    """Van der Eijk's A: Rescaled to 0 (Agreement) to 1 (Polarization)."""
    S = np.count_nonzero(freqs)
    K = len(freqs)
    if K <= 1: return 0
    A = 1 - (S - 1) / (K - 1) 
    return (1 - A) / 2 

def get_balance(data):
    c1, c2 = np.sum(data < 0), np.sum(data >= 0)
    if max(c1, c2) == 0: return 0
    return min(c1, c2) / max(c1, c2)

# DATA GENERATION
np.random.seed(42)
n = 10000

# Test 1: Skewed Unimodal (Exponential decay)
data_uni = (np.random.exponential(0.2, n) / 5) * 2 - 1
freqs_uni, _ = np.histogram(data_uni, bins=10, range=(-1, 1), density=True)

# Test 2: Lopsided Bimodal (85% agree, 15% strong dissent)
m1, m2 = np.random.normal(-0.7, 0.1, 8500), np.random.normal(0.7, 0.1, 1500)
data_bi = np.clip(np.concatenate([m1, m2]), -1, 1)
freqs_bi, _ = np.histogram(data_bi, bins=10, range=(-1, 1), density=True)



def run_experiment(name, data, freqs):
    print(f"\n--- {name} ---")
    print(f"{'Metric':<20} | {'Score':<8} | {'Verdict'}")
    print("-" * 50)
    print(f"{'Sarle BC':<20} | {get_sarle_bc(data):.4f} | {'FAIL' if get_sarle_bc(data) > 0.55 else 'PASS'}")
    print(f"{'DFU':<20} | {get_dfu(freqs):.4f} | {'PASS' if (get_dfu(freqs) > 0) == ('BIMODAL' in name) else 'FAIL'}")
    print(f"{'Agreement (A)':<20} | {get_agreement_a(freqs):.4f} | {'Spatial-dependent'}")
    print(f"{'Balance':<20} | {get_balance(data):.4f} | {'FAIL' if get_balance(data) < 0.2 and 'BIMODAL' in name else 'PASS'}")

run_experiment("SKEWED UNIMODAL", data_uni, freqs_uni)
run_experiment("LOPSIDED BIMODAL", data_bi, freqs_bi)