import matplotlib.pyplot as plt

def plot_results(stab_a, stab_b, stab_c, stab_d, w_a, w_b, w_c, w_d):
    """
    Zeigt die Stabilitätsverläufe und die Entwicklung der self_eval-Gewichte.
    """
    plt.figure(figsize=(14, 10))

    plt.subplot(4, 1, 1)
    plt.plot(stab_a, label="System A: Prime")
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(stab_b, label="System B: A-basiert")
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(stab_c, label="System C: Zufall")
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(stab_d, label="System D: Muster 1011")
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(w_a, label="Gewicht A (self_eval)")
    plt.plot(w_b, label="Gewicht B (self_eval)")
    plt.plot(w_c, label="Gewicht C (self_eval)")
    plt.plot(w_d, label="Gewicht D (self_eval)")
    plt.title("self_eval-Gewichte über Zeit")
    plt.legend()
    plt.tight_layout()
    plt.show()


def write_to_file(filename, label, states, stabilities, weights):
    """
    Schreibt die Systemhistorie in eine Textdatei.
    """
    with open(filename, "a") as f:
        f.write(f"System {label}\n")
        f.write(f"{'Step':>6} | {'State':>5} | {'Stability':>9} | {'Self_Eval':>9}\n")
        f.write("-" * 36 + "\n")
        for i in range(len(states)):
            f.write(f"{i:6d} | {states[i]:5d} | {stabilities[i]:9.4f} | {weights[i]:9.4f}\n")
        f.write("\n")