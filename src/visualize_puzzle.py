import networkx as nx
import matplotlib.pyplot as plt
from puzzle_generator import generate_single_path

def visualize_puzzle_graph(G: nx.DiGraph, save_path: str = "puzzle_graph.png"):
    """Visualize the puzzle graph with node labels and edge transitions"""
    plt.figure(figsize=(12, 8))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=2000, alpha=0.7)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                          arrows=True, arrowsize=20)
    
    # Add node labels (strings at each node)
    node_labels = nx.get_node_attributes(G, 'string')
    nx.draw_networkx_labels(G, pos, node_labels, font_size=10)
    
    # Add edge labels (transition rules)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
    plt.title("Puzzle Solution Graph")
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    # Generate a single puzzle with t=3 transitions and d=5 steps
    G, root, transitions, transition_history = generate_single_path(n=3, t=3, d=5)
    
    # Print puzzle details
    print("Initial string:", root)
    print("\nTransitions:")
    for i, trans in enumerate(transitions):
        print(f"{i}: {trans['src']} -> {trans['tgt']}")
    print("\nSolution sequence:", transition_history)
    
    # Visualize and save the graph
    visualize_puzzle_graph(G)
    print("\nGraph saved as 'puzzle_graph.png'")