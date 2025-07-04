import matplotlib.pyplot as plt
from recuria.engine import run_system, run_system_d
from recuria.primes import generate_prime_input, normalize_stability, generate_random_input, generate_repeating_pattern

def run_recuria_on_model(max_steps, output_path):
    init_a = {'input': 0.5, 'state': 0.3, 'self_eval': 0.2}
    init_b = {'input': 0.6, 'state': 0.2, 'self_eval': 0.2}
    init_c = {'input': 0.9, 'state': 0.05, 'self_eval': 0.05}
    init_d = {'input': 0.7, 'state': 0.15, 'self_eval': 0.15}

    input_a = generate_prime_input(max_steps)
    states_a, stab_a, w_a = run_system(input_a, init_a)

    input_b = normalize_stability(stab_a)
    states_b, stab_b, w_b = run_system(input_b, init_b)

    input_c = generate_random_input(max_steps)
    states_c, stab_c, w_c = run_system(input_c, init_c)

    pattern_input_d = generate_repeating_pattern(max_steps, [1,0,1,1])
    states_d, stab_d, w_d = run_system_d(pattern_input_d, init_d)

    # Visualisierung
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

    # Ergebnisse speichern
    with open(output_path, "a") as f:
        for label, states, stab, weights in zip(["A", "B", "C", "D"],
                                                [states_a, states_b, states_c, states_d],
                                                [stab_a, stab_b, stab_c, stab_d],
                                                [w_a, w_b, w_c, w_d]):
            f.write(f"System {label}\n")
            f.write(f"{'Step':>6} | {'State':>5} | {'Stability':>9} | {'Self_Eval':>9}\n")
            f.write("-" * 36 + "\n")
            for i in range(len(states)):
                f.write(f"{i:6d} | {states[i]:5d} | {stab[i]:9.4f} | {weights[i]:9.4f}\n")
            f.write("\n")