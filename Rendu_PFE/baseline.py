import networkx as nx
import subprocess
import random
from blackbox_interface import get_path_cost_with_blackbox

def generate_random_path(G, start, target):
    """
    Génère un chemin aléatoire entre un noeud de départ et une cible en respectant la connectivité.
    """
    path = [start]
    current_node = start
    previous_node = None 

    while current_node != target:
        neighbors = list(G.neighbors(current_node))
        if not neighbors:
            raise ValueError(f"Aucun voisin trouvé pour le nœud {current_node}, chemin impossible.")

        if previous_node in neighbors and len(neighbors) > 1:
            neighbors.remove(previous_node)

        if target in neighbors:  
            path.append(target)
            break

        next_node = random.choice(neighbors)
        path.append(next_node)
        previous_node = current_node
        current_node = next_node

    return path

def compute_num_random_paths(num_nodes, density, alpha=0.1, beta=0.5):
    """
    Calcule dynamiquement le nombre de chemins aléatoires à générer
    selon la taille du graphe et sa densité.
    """
    return max(1, int(alpha * num_nodes + beta * density * num_nodes) + 1)



def baseline_method(G, source, target, seed, blackbox_path, density=None, num_random_paths=None):
    """
    Implémente la méthode témoin pour trouver un chemin de coût minimal.
    Si num_random_paths n'est pas fourni, il est calculé dynamiquement.
    """
    num_nodes = len(G.nodes())

    # Calcul dynamique du nombre de chemins aléatoires
    if num_random_paths is None:
        if density is None:
            # Si la densité n'est pas fournie, on l'estime à partir du graphe
            possible_edges = num_nodes * (num_nodes - 1) / 2
            density = len(G.edges()) / possible_edges
        num_random_paths = compute_num_random_paths(num_nodes, density)

    print(f"[INFO] Nombre de chemins aléatoires générés par voisin : {num_random_paths}")

    current_node = source
    previous_node = None
    path = [source]
    total_cost = 0

    while current_node != target:
        neighbors = list(G.neighbors(current_node))
        if not neighbors:
            raise ValueError(f"Aucun chemin vers la cible {target} depuis le nœud {current_node}")

        if previous_node in neighbors and len(neighbors) > 1:
            neighbors.remove(previous_node)

        if len(neighbors) == 1:
            next_node = neighbors[0]
            path.append(next_node)
            total_cost = get_path_cost_with_blackbox(seed, path, blackbox_path)
            previous_node = current_node
            current_node = next_node
            continue

        if target in neighbors:
            path.append(target)
            total_cost = get_path_cost_with_blackbox(seed, path, blackbox_path)
            break

        min_cost = float('inf')
        best_next_node = None

        for neighbor in neighbors:
            random_path = generate_random_path(G, neighbor, target)
            cost = get_path_cost_with_blackbox(seed, random_path, blackbox_path)
            if cost < min_cost:
                min_cost = cost
                best_next_node = neighbor

        path.append(best_next_node)
        total_cost = min_cost
        previous_node = current_node
        current_node = best_next_node

    return path, total_cost
