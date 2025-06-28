# ------------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------



import random
import matplotlib.pyplot as plt

# Maximum number of simulation steps
# (up to 1,000,000 possible with precomputed primes loaded for faster computation)
MAX_STEPS = 10000
# Size of memory (number of past states considered)
MEMORY_SIZE = 200
# Output file for simulation results
OUTPUT_FILE = "system_output.txt"

# --- Precomputed primes up to 1 million ---
# This file 'primes_to_1M.txt' must contain a list of primes,
# one per line or separated by whitespace
with open("primes_to_1M.txt") as f:
    PRECOMPUTED_PRIMES = set(map(int, f.read().split()))

# --- Helper functions ---

def is_prime(n):
    """
    Checks if a number n is prime by verifying membership in
    the precomputed prime set PRECOMPUTED_PRIMES.
    """
    return n in PRECOMPUTED_PRIMES

def weighted_sum(input_signal, state, self_eval, weights):
    """
    Calculates the weighted sum of the three input values:
    current input, previous state, and self-evaluation.
    """
    return input_signal * weights['input'] + state * weights['state'] + self_eval * weights['self_eval']

def update_weights(weights, stability, threshold=0.8, step=0.05):
    """
    Dynamically adjusts the 'self_eval' weight:
    - Increases weight (up to 1.0) if stability is above threshold
    - Decreases weight (down to 0.0) if stability is below threshold
    """
    if stability > threshold:
        weights['self_eval'] = min(1.0, weights['self_eval'] + step)
    else:
        weights['self_eval'] = max(0.0, weights['self_eval'] - step)

def average(memory):
    """
    Calculates the average value of past states stored in memory.
    Returns 0 if memory is empty.
    """
    return sum(memory) / len(memory) if memory else 0

def calculate_stability(memory):
    """
    Calculates stability as the proportion of consecutive equal values
    in memory (e.g. how often the state remained the same).
    """
    if len(memory) < 2:
        return 1.0
    stable = sum(1 for i in range(1, len(memory)) if memory[i] == memory[i - 1])
    return stable / (len(memory) - 1)

def normalize_stability(stability_list, threshold=0.8):
    """
    Converts a list of stability values into binary values:
    1 if stability >= threshold, otherwise 0.
    """
    return [1 if s >= threshold else 0 for s in stability_list]

def generate_prime_input(max_steps):
    """
    Generates an input sequence with 1 for primes and 0 otherwise,
    starting from 2 up to 2 + max_steps.
    """
    return [1 if is_prime(n) else 0 for n in range(2, 2 + max_steps)]

def generate_random_input(max_steps):
    """
    Generates a random binary input sequence of length max_steps.
    """
    return [random.choice([0,1]) for _ in range(max_steps)]

def generate_repeating_pattern(length, pattern):
    """
    Generates an input sequence of length 'length' that contains a
    repeating pattern 'pattern'.
    """
    return [pattern[i % len(pattern)] for i in range(length)]

# --- System functions ---

def run_system(input_sequence, init_weights):
    """
    Simulates the recursive system with a given input sequence and initial weights.
    Returns the histories of states, stability, and self_eval weights.
    """
    state = 0
    memory = []
    weights = init_weights.copy()
    states_history, stability_history, weight_history = [], [], []

    for input_signal in input_sequence:
        self_eval = average(memory)
        sum_value = weighted_sum(input_signal, state, self_eval, weights)
        decision = 1 if sum_value >= 0.5 else 0
        state = decision

        memory.append(state)
        # Limit memory size to MEMORY_SIZE
        if len(memory) > MEMORY_SIZE:
            memory.pop(0)

        stability = calculate_stability(memory)
        update_weights(weights, stability)

        states_history.append(state)
        stability_history.append(stability)
        weight_history.append(weights['self_eval'])

    return states_history, stability_history, weight_history

def run_system_d(input_sequence, init_weights):
    """
    System D uses a predefined input with a pattern (e.g. 1011),
    to observe stability on repeating patterns.
    Otherwise functions identically to run_system.
    """
    state = 0
    memory = []
    weights = init_weights.copy()
    states_history, stability_history, weight_history = [], [], []

    for input_signal in input_sequence:
        self_eval = average(memory)
        sum_value = weighted_sum(input_signal, state, self_eval, weights)
        decision = 1 if sum_value >= 0.5 else 0
        state = decision

        memory.append(state)
        if len(memory) > MEMORY_SIZE:
            memory.pop(0)

        stability = calculate_stability(memory)
        update_weights(weights, stability)

        states_history.append(state)
        stability_history.append(stability)
        weight_history.append(weights['self_eval'])

    return states_history, stability_history, weight_history

def plot_results(stab_a, stab_b, stab_c, stab_d, w_a, w_b, w_c, w_d):
    """
    Visualizes the stability of the four systems and their self_eval weights.
    """
    plt.figure(figsize=(14, 10))
    plt.subplot(4,1,1)
    plt.plot(stab_a, label="System A Stability (Primes)")
    plt.legend()

    plt.subplot(4,1,2)
    plt.plot(stab_b, label="System B Stability (Dependent on A)")
    plt.legend()

    plt.subplot(4,1,3)
    plt.plot(stab_c, label="System C Stability (Random)")
    plt.legend()

    plt.subplot(4,1,4)
    plt.plot(stab_d, label="System D Stability (Pattern 1011)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10,4))
    plt.plot(w_a, label="Weight A (self_eval)")
    plt.plot(w_b, label="Weight B (self_eval)")
    plt.plot(w_c, label="Weight C (self_eval)")
    plt.plot(w_d, label="Weight D (self_eval)")
    plt.title("self_eval Weights Over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()

def write_to_file(filename, label, states, stabilities, weights):
    """
    Writes the results to a text file with formatted columns for easier reading.
    """
    with open(filename, "a") as f:
        f.write(f"System {label}\n")
        f.write(f"{'Step':>6} | {'State':>5} | {'Stability':>9} | {'Self_Eval':>9}\n")
        f.write("-" * 36 + "\n")
        for i in range(len(states)):
            f.write(f"{i:6d} | {states[i]:5d} | {stabilities[i]:9.4f} | {weights[i]:9.4f}\n")
        f.write("\n")

# --- Main execution ---

if __name__ == "__main__":
    # Individual initial weights for each system
    init_a = {'input': 0.5, 'state': 0.3, 'self_eval': 0.2}
    init_b = {'input': 0.6, 'state': 0.2, 'self_eval': 0.2}
    init_c = {'input': 0.9, 'state': 0.05, 'self_eval': 0.05}
    init_d = {'input': 0.7, 'state': 0.15, 'self_eval': 0.15}

    # Generate inputs for each system
    input_a = generate_prime_input(MAX_STEPS)               # Prime numbers input
    states_a, stab_a, w_a = run_system(input_a, init_a)     # Simulate System A

    input_b = normalize_stability(stab_a)                   # Input for System B from stability of A
    states_b, stab_b, w_b = run_system(input_b, init_b)     # Simulate System B

    input_c = generate_random_input(MAX_STEPS)              # Random input
    states_c, stab_c, w_c = run_system(input_c, init_c)     # Simulate System C

    pattern_input_d = generate_repeating_pattern(MAX_STEPS, [1, 0, 1, 1])  # Pattern 1011
    states_d, stab_d, w_d = run_system_d(pattern_input_d, init_d)           # Simulate System D

    # Plot results
    plot_results(stab_a, stab_b, stab_c, stab_d, w_a, w_b, w_c, w_d)

    # Write results to file
    write_to_file(OUTPUT_FILE, "A", states_a, stab_a, w_a)
    write_to_file(OUTPUT_FILE, "B", states_b, stab_b, w_b)
    write_to_file(OUTPUT_FILE, "C", states_c, stab_c, w_c)
    write_to_file(OUTPUT_FILE, "D", states_d, stab_d, w_d)