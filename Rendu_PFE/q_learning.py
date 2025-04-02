import numpy as np
import random
import time
from tqdm import tqdm
from blackbox_interface import get_path_cost_with_blackbox
import networkx as nx

class GraphEnvironment:
    """
    Environnement représentant un graphe, où un agent se déplace de la source à la cible.
    """
    def __init__(self, graph, start, goal, seed, blackbox_path):
        """
        Initialise l'environnement du graphe avec la source, la cible et les paramètres nécessaires.
        """
        self.graph = graph
        self.start = start
        self.goal = goal
        self.current_node = start
        self.path = [start]
        self.visited_nodes = set([start])
        self.seed = seed
        self.blackbox_path = blackbox_path
        self.revisit_penalty = 50
        self.overlength_penalty = 100

    def reset(self):
        """
        Réinitialise l'environnement et renvoie l'état de départ.
        """
        self.current_node = self.start
        self.path = [self.start]
        self.visited_nodes = set([self.start])
        return self.current_node

    def step(self, action):
        """
        Exécute une action et retourne l'état suivant, la récompense et un booléen indiquant la fin de l'épisode.
        """
        if action not in self.graph[self.current_node]:
            raise ValueError(f"Action {action} is not valid from node {self.current_node}")

        try:
            remaining_hops = nx.shortest_path_length(self.graph, source=action, target=self.goal)
        except nx.NetworkXNoPath:
            remaining_hops = float('inf')

        if len(self.path) + remaining_hops > len(self.graph.nodes):
            return self.current_node, -self.overlength_penalty, True

        if action in self.visited_nodes:
            return self.current_node, -self.revisit_penalty, True

        self.path.append(action)
        self.visited_nodes.add(action)

        if action == self.goal:
            cost = get_path_cost_with_blackbox(self.seed, self.path, self.blackbox_path)
            reward = 1 / cost if cost else -100
            return action, reward, True

        self.current_node = action
        return self.current_node, 0, False

    def get_actions(self, node):
        """
        Retourne la liste des actions possibles (voisins) depuis un nœud donné.
        """
        return list(self.graph.neighbors(node))

def q_learning(env, duration, blackbox_path, alpha=0.2, gamma=0.9, epsilon_start=0.9, epsilon_end=0.1):
    """
    Exécute l'algorithme de Q-Learning sur un environnement de graphe pendant une durée donnée.
    """
    Q = {node: {action: 0 for action in env.get_actions(node)} for node in env.graph.nodes()}
    total_rewards_per_episode = []
    start_time = time.time()

    with tqdm(desc="Entraînement Q-Learning en cours") as pbar:
        while time.time() - start_time < duration:
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * ((time.time() - start_time) / duration)
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                if not env.get_actions(state):
                    break

                if np.random.rand() < epsilon:
                    action = random.choice(env.get_actions(state))
                else:
                    action = max(Q[state], key=Q[state].get, default=random.choice(env.get_actions(state)))

                next_state, reward, done = env.step(action)

                if next_state not in Q:
                    Q[next_state] = {a: 0 for a in env.get_actions(next_state)}

                best_next_value = max(Q[next_state].values(), default=0)
                Q[state][action] += alpha * (reward + gamma * best_next_value - Q[state][action])

                total_reward += reward
                state = next_state

            total_rewards_per_episode.append(total_reward)
            pbar.update(1)
    
    return Q

def evaluate_policy(env, Q, blackbox_path):
    """
    Évalue la politique extraite de la table Q en suivant les actions maximisant la valeur.
    """
    state = env.reset()
    path = [state]
    done = False

    while not done:
        if state not in Q or not Q[state]:
            break

        action = max(Q[state], key=Q[state].get)
        if action in path:
            break

        next_state, _, done = env.step(action)
        path.append(next_state)
        state = next_state

    if state == env.goal:
        real_cost = get_path_cost_with_blackbox(env.seed, path, blackbox_path)
    else:
        real_cost = None

    return path, real_cost
