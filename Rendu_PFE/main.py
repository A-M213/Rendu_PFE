import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from graph_utils import generate_connected_graph
from baseline import baseline_method
from monte_carlo import (
    monte_carlo_simulation,
    monte_carlo_simulation_with_exploration,
    monte_carlo_with_nested_rollouts
)
from q_learning import GraphEnvironment, q_learning, evaluate_policy

graph_sizes = [50, 100]
graph_densities = [0.3, 0.6]
durations = [5, 10]
seed = 0
blackbox_path = "h:/Desktop/pfe/blackBox.exe"
num_graphs = 2

results = []

for num_nodes in graph_sizes:
    for density in graph_densities:
        for duration in durations:
            baseline_costs = []
            mc_costs = []
            mc_exploration_costs = []
            mc_nested_costs = []
            q_learning_costs = []

            for _ in range(num_graphs):
                graph = generate_connected_graph(num_nodes, density)
                source, target = 0, num_nodes - 1

                _, b_cost = baseline_method(graph, source, target, seed, blackbox_path)
                baseline_costs.append(b_cost)

                _, mc_cost = monte_carlo_simulation(graph, source, target, duration, seed, blackbox_path)
                mc_costs.append(mc_cost)

                _, mc_exp_cost = monte_carlo_simulation_with_exploration(
                    graph, source, target, duration, seed, blackbox_path
                )
                mc_exploration_costs.append(mc_exp_cost)

                _, mc_nested_cost = monte_carlo_with_nested_rollouts(
                    graph, source, target, duration, seed, blackbox_path
                )
                mc_nested_costs.append(mc_nested_cost)

                env = GraphEnvironment(graph, source, target, seed, blackbox_path)
                q_table = q_learning(env, duration, blackbox_path)
                _, q_cost = evaluate_policy(env, q_table, blackbox_path)
                q_learning_costs.append(q_cost)

            avg_baseline_cost = np.mean(baseline_costs)
            avg_mc_cost = np.mean(mc_costs)
            avg_mc_exploration_cost = np.mean(mc_exploration_costs)
            avg_mc_nested_cost = np.mean(mc_nested_costs)
            avg_q_learning_cost = np.mean(q_learning_costs)

            result_str = (
                f"\n=== Résultats pour Graphe={num_nodes}, Densité={density}, Duration={duration} ===\n"
                f"Baseline : Coût moyen {avg_baseline_cost:.4f}\n"
                f"Monte Carlo : Coût moyen {avg_mc_cost:.4f}\n"
                f"Monte Carlo Exploration : Coût moyen {avg_mc_exploration_cost:.4f}\n"
                f"Monte Carlo Nested Rollouts : Coût moyen {avg_mc_nested_cost:.4f}\n"
                f"Q-Learning : Coût moyen {avg_q_learning_cost:.4f}\n"
            )
            results.append(result_str)
            print(result_str)

            methods = [
                "Baseline",
                "MC",
                "MC+Exploration",
                "MC+Nested",
                "Q-Learning"
            ]
            costs = [
                avg_baseline_cost,
                avg_mc_cost,
                avg_mc_exploration_cost,
                avg_mc_nested_cost,
                avg_q_learning_cost
            ]
            
            plt.figure(figsize=(8, 6))
            plt.bar(methods, costs, color=['blue','orange','green','red','purple'])
            plt.title(f"Graphe={num_nodes}, Densité={density}, Durée={duration}")
            plt.ylabel("Coût moyen final")
            plt.xlabel("Méthodes")
            plt.ylim([0, max(costs)*1.2])
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)

            plt.savefig(f"bar_G{num_nodes}_Dens{density}_Dur{duration}.png")
            plt.close()

output_file = "execution_results_test.txt"
with open(output_file, "w") as f:
    f.writelines(results)

print(f"\nLes résultats ont été sauvegardés dans {output_file}")

def generate_convergence_plots_from_file(filename="execution_results_test.txt"):
    """
    Genère courbe de convergence.
    """
    import re
    from collections import defaultdict

    pattern = re.compile(r"Graphe=(\d+), Densité=([0-9.]+), Duration=(\d+)")
    cost_pattern = re.compile(r"(.+) : Coût moyen ([0-9.]+)")

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    durations = set()

    with open(filename, "r") as f:
        lines = f.readlines()

    current = None
    for line in lines:
        header_match = pattern.search(line)
        if header_match:
            graphe, dens, dur = int(header_match[1]), float(header_match[2]), int(header_match[3])
            current = (graphe, dens, dur)
            durations.add(dur)
        elif current:
            cost_match = cost_pattern.search(line)
            if cost_match:
                method, cost = cost_match[1], float(cost_match[2])
                data[current[0]][current[1]][current[2]][method.strip()] = cost

    methods = [
        "Baseline",
        "Monte Carlo",
        "Monte Carlo Exploration",
        "Monte Carlo Nested Rollouts",
        "Q-Learning"
    ]
    colors = {
        "Baseline": "black",
        "Monte Carlo": "tab:blue",
        "Monte Carlo Exploration": "tab:orange",
        "Monte Carlo Nested Rollouts": "tab:green",
        "Q-Learning": "tab:red"
    }

    for graphe in sorted(data.keys()):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for i, dens in enumerate([0.3, 0.6]):
            ax = axes[i]
            for method in methods:
                y = []
                for dur in sorted(durations):
                    y.append(data[graphe][dens].get(dur, {}).get(method, None))
                ax.plot(sorted(durations), y, marker='o', label=method, color=colors[method])
            ax.set_title(f'Courbe de convergence - {graphe} nœuds (densité {dens})')
            ax.set_xlabel("Temps d'entraînement (s)")
            ax.set_ylabel("Coût")
            ax.grid(True)
            ax.legend()
        fig.tight_layout()
        plt.savefig(f"courbe_convergence_{graphe}_noeuds.png")
        plt.close()

generate_convergence_plots_from_file()
