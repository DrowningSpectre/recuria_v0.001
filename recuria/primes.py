import random

# Vorausgesetzte Menge von Primzahlen, hier beispielhaft als leeres Set, 
# du solltest deine primes_to_1M.txt laden und füllen.
PRECOMPUTED_PRIMES = set()

def is_prime(n):
    return n in PRECOMPUTED_PRIMES

def generate_prime_input(max_steps):
    """
    Gibt eine Liste zurück, die für jeden Wert von 2 bis 2 + max_steps
    eine 1 enthält, wenn die Zahl eine Primzahl ist, sonst 0.
    """
    return [1 if is_prime(n) else 0 for n in range(2, 2 + max_steps)]

def normalize_stability(stability_list, threshold=0.8):
    """
    Wandelt eine Liste von Stabilitätswerten in binäre Werte um:
    1 wenn Stabilität >= threshold, sonst 0.
    """
    return [1 if s >= threshold else 0 for s in stability_list]

def generate_random_input(max_steps):
    """
    Erzeugt eine zufällige Binärsequenz der Länge max_steps.
    """
    return [random.choice([0, 1]) for _ in range(max_steps)]

def generate_repeating_pattern(length, pattern):
    """
    Erzeugt eine Liste der Länge 'length', die das Muster 'pattern' 
    periodisch wiederholt.
    """
    return [pattern[i % len(pattern)] for i in range(length)]