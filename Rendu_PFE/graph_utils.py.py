import networkx as nx
import matplotlib.pyplot as plt

def generate_connected_graph(num_nodes, probability):
    """
    Génère un graphe connexe avec un nombre donné de sommets et une probabilité d'avoir une arête.
    Réessaie jusqu'à ce qu'un graphe connexe soit créé.
    """
    while True:
        G = nx.gnp_random_graph(num_nodes, probability)
        if nx.is_connected(G):
            print(f"Graphe connexe généré avec {num_nodes} sommets et une densité de {probability}.")
            return G
        print("Graphe non connexe détecté, tentative de régénération...")

def draw_graph(graph):
    """
    Affiche le graphe avec Matplotlib.
    """
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph)  # Disposition des nœuds
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.title("Visualisation du Graphe")
    plt.show()

# Exemple d'utilisation
#graph = generate_connected_graph(10, 0.2)
#draw_graph(graph)