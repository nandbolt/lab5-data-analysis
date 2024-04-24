import numpy as np

# Given values
a = 4.94 # m
f = 4.71238 # m
b = 9.564 # m
freqs = [200, 284, 372, 491, 639, 783, 908] # rev/s
for i in range(len(freqs)):
    freqs[i] = freqs[i] * 2 * np.pi
x = [0.0011362449667242462, 0.0016206754248853671, 0.0021490832631714915, 0.002950637378027666, 0.00393033442308869, 0.004895286015736452, 0.005547528739234263] # m
sigma_w = 6.28 # rad/s
sigma_a = sigma_f = .001 # mm converted to meters
sigma_b = .002 # mm converted to meters
sigma_x = .0005 # mm converted to meters

# Calculate w from freqs (convert rev/s to rad/s)
w = [2 * np.pi * freq for freq in freqs]

# Function to calculate c
def calculate_c(w, a, f, b, x):
    return 4 * w * a * (f + b) / x

# Partial derivatives
def partial_w(a, f, b, x):
    return 4 * a * (f + b) / x

def partial_a(w, f, x):
    return 4 * w * f / x

def partial_f(w, a, x):
    return 4 * w * a / x

def partial_b(w, a, x):
    return 4 * w * a / x

def partial_x(w, a, f, b, x):
    return -4 * w * a * (f + b) / (x ** 2)

# Calculate uncertainty in c
def calculate_sigma_c(w, a, f, b, x, sigma_w, sigma_a, sigma_f, sigma_b, sigma_x):
    term1 = (partial_w(a, f, b, x) * sigma_w) ** 2
    term2 = (partial_a(w, f, x) * sigma_a) ** 2
    term3 = (partial_f(w, a, x) * sigma_f) ** 2
    term4 = (partial_b(w, a, x) * sigma_b) ** 2
    term5 = (partial_x(w, a, f, b, x) * sigma_x) ** 2
    return np.sqrt(term1 + term2 + term3 + term4 + term5)

# Calculate c and its uncertainty for each set of data
for i in range(len(freqs)):
    c = calculate_c(w[i], a, f, b, x[i])
    sigma_c = calculate_sigma_c(w[i], a, f, b, x[i], sigma_w, sigma_a, sigma_f, sigma_b, sigma_x)
    print(f"For data point {i+1}:")
    print(f"c = {c:.2f} m/s, σ_c = {sigma_c:.2f} m/s")