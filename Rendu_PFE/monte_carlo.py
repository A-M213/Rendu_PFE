import random
import time
from collections import defaultdict
from blackbox_interface import get_path_cost_with_blackbox

def generate_random_path(G, source, target):
    """
    Génère un chemin aléatoire simple entre la source et la cible en évitant les revisites immédiates.
    """
    path = [source]
    current = source
    visited = set(path)
    while current != target:
        neighbors = list(set(G.neighbors(current)) - visited)
        if not neighbors:
            return None
        next_node = random.choice(neighbors)
        path.append(next_node)
        visited.add(next_node)
        current = next_node
    return path

def generate_random_path_weighted(G, source, target, memory):
    """
    Génère un chemin pondéré où les transitions avec moins de visites sont favorisées.
    """
    path = [source]
    current = source
    visited = set(path)
    while current != target:
        neighbors = list(set(G.neighbors(current)) - visited)
        if not neighbors:
            return None
        weights = [1 / (memory[(current, neighbor)] + 1e-6) for neighbor in neighbors]
        next_node = random.choices(neighbors, weights=weights, k=1)[0]
        path.append(next_node)
        visited.add(next_node)
        current = next_node
    return path

def monte_carlo_simulation(G, source, target, duration, seed, blackbox_path):
    """
    Monte Carlo classique : teste des chemins aléatoires et sélectionne le meilleur.
    Affiche uniquement lorsqu'un nouveau meilleur chemin est trouvé.
    À la fin, affiche le nombre total d'itérations (exécutions) réalisées.
    """
    best_path = None
    best_cost = float('inf')
    start_time = time.time()
    iteration = 0

    while time.time() - start_time < duration:
        iteration += 1
        path = generate_random_path(G, source, target)
        if path:
            cost = get_path_cost_with_blackbox(seed, path, blackbox_path)
            if cost is not None and cost < best_cost:
                best_cost = cost
                best_path = path
                print(f"[Classique] it={iteration}, meilleur coût={best_cost}, chemin={best_path}")

    print(f"[Classique] Nombre total d'itérations exécutées : {iteration}")
    return best_path, best_cost

def monte_carlo_simulation_with_exploration(G, source, target, duration, seed, blackbox_path):
    """
    Monte Carlo amélioré : privilégie l'exploration en utilisant une mémoire de visites,
    et pénalise davantage les chemins de coût élevé (pénalité = 1 + cost * alpha).
    Affiche uniquement lorsqu'un nouveau meilleur chemin est trouvé,
    puis affiche à la fin le nombre total d'itérations réalisées.
    """
    from collections import defaultdict
    memory = defaultdict(lambda: 1.0)
    best_path = None
    best_cost = float('inf')
    start_time = time.time()

    alpha = 100
    iteration = 0

    while time.time() - start_time < duration:
        iteration += 1
        path = generate_random_path_weighted(G, source, target, memory)
        if path:
            cost = get_path_cost_with_blackbox(seed, path, blackbox_path)
            if cost is not None:
                penalty = 1 + cost * alpha
                for i in range(len(path) - 1):
                    memory[(path[i], path[i+1])] += penalty

                if cost < best_cost:
                    best_cost = cost
                    best_path = path
                    print(f"[Exploration-MODIF] it={iteration}, meilleur coût={best_cost}, chemin={best_path}")

    print(f"[Exploration-MODIF] Nombre total d'itérations exécutées : {iteration}")
    return best_path, best_cost

def monte_carlo_with_nested_rollouts(G, source, target, duration, seed, blackbox_path, depth=3):
    """
    Monte Carlo avec Nested Rollouts amélioré :
      1) Pas de 'return' immédiat si neighbor == target : on compare le coût.
      2) Vérification du temps global 'time.time() - start_time' à chaque étape,
         pour ne pas dépasser 'duration'.
      3) Paramètres simplifiés dans la récursion (seed retiré).
      4) Fallback aléatoire si depth == 0 (pas de profondeur).
      5) On compare tous les chemins menant à la cible pour garder le meilleur.
      6) Un compteur 'eval_count' incrémente à chaque évaluation d'un chemin complet.
    """
    import random, time
    from collections import defaultdict
    random.seed(seed)

    best_path = None
    best_cost = float('inf')
    start_time = time.time()
    eval_count = 0
    main_iterations = 0

    def extend_randomly_to_target(current_path, max_fallback_length=50):
        path = current_path[:]
        visited = set(path)
        current_node = path[-1]
        steps = 0

        while current_node != target:
            if time.time() - start_time >= duration:
                return None
            if steps > max_fallback_length:
                return None
            neighbors = list(set(G.neighbors(current_node)) - visited)
            if not neighbors:
                return None
            next_node = random.choice(neighbors)
            path.append(next_node)
            visited.add(next_node)
            current_node = next_node
            steps += 1
        return path

    def nested_rollout(current_path, depth_local):
        nonlocal eval_count
        if time.time() - start_time >= duration:
            return None
        if current_path[-1] == target:
            return current_path
        if depth_local == 0:
            return extend_randomly_to_target(current_path)

        current_node = current_path[-1]
        neighbors = list(G.neighbors(current_node))
        random.shuffle(neighbors)

        best_cost_local = float('inf')
        best_path_local = None

        for neighbor in neighbors:
            if time.time() - start_time >= duration:
                break
            if neighbor not in current_path:
                new_path = current_path + [neighbor]
                candidate_path = new_path if neighbor == target else nested_rollout(new_path, depth_local - 1)
                if candidate_path and candidate_path[-1] == target:
                    eval_count += 1
                    cost_candidate = get_path_cost_with_blackbox(seed, candidate_path, blackbox_path)
                    if cost_candidate is not None and cost_candidate < best_cost_local:
                        best_cost_local = cost_candidate
                        best_path_local = candidate_path
        return best_path_local

    while time.time() - start_time < duration:
        main_iterations += 1
        path_candidate = nested_rollout([source], depth)
        if path_candidate and path_candidate[-1] == target:
            cost = get_path_cost_with_blackbox(seed, path_candidate, blackbox_path)
            if cost is not None and cost < best_cost:
                best_cost = cost
                best_path = path_candidate
                print(f"[NestedRollouts-Improved] main_it={main_iterations}, eval_count={eval_count}, meilleur coût={best_cost}, chemin={best_path}")

    print(f"[NestedRollouts-Improved] Nombre total d'itérations principales : {main_iterations}")
    print(f"[NestedRollouts-Improved] Nombre total d'évaluations de chemins candidats : {eval_count}")
    return best_path, best_cost
