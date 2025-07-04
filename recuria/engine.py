import random

MEMORY_SIZE = 10  # Maximale Größe des Speicherfensters

def average(memory):
    """Berechnet den Durchschnitt der gespeicherten Zustände."""
    return sum(memory) / len(memory) if memory else 0

def weighted_sum(input_signal, state, self_eval, weights):
    """Berechnet die gewichtete Summe der Eingaben."""
    return input_signal * weights['input'] + state * weights['state'] + self_eval * weights['self_eval']

def calculate_stability(memory):
    """
    Berechnet die Stabilität als Anteil der aufeinanderfolgenden gleichen Zustände.
    Gibt 1.0 zurück, wenn Memory < 2 (kein Vergleich möglich).
    """
    if len(memory) < 2:
        return 1.0
    stable = sum(1 for i in range(1, len(memory)) if memory[i] == memory[i - 1])
    return stable / (len(memory) - 1)

def update_weights(weights, stability, threshold=0.8, step=0.05):
    """
    Passt das Gewicht 'self_eval' dynamisch an:
    - Erhöht Gewicht, wenn Stabilität über Threshold liegt
    - Verringert Gewicht, wenn Stabilität darunter liegt
    """
    if stability > threshold:
        weights['self_eval'] = min(1.0, weights['self_eval'] + step)
    else:
        weights['self_eval'] = max(0.0, weights['self_eval'] - step)

def run_system(input_sequence, init_weights):
    """
    Simuliert das rekursive System mit gegebener Eingabesequenz und Startgewichten.
    Gibt Historien von Zuständen, Stabilitäten und 'self_eval' Gewichten zurück.
    """
    state = 0
    memory = []
    weights = init_weights.copy()
    states_history = []
    stability_history = []
    weight_history = []

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

def run_system_d(input_sequence, init_weights):
    """
    Variante des Systems, die speziell für periodische Muster (z.B. 1011) gedacht ist.
    Ansonsten arbeitet sie identisch zu run_system.
    """
    state = 0
    memory = []
    weights = init_weights.copy()
    states_history = []
    stability_history = []
    weight_history = []

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