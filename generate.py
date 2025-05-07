import numpy as np
import networkx as nx
import scipy.sparse as ssp
import os
import argparse
from tqdm import tqdm
from community import community_louvain
import random


def generate_ba_network(n, m):
    G = nx.barabasi_albert_graph(n, m)
    return nx.to_directed(G)


def generate_er_network(n, p):
    G = nx.erdos_renyi_graph(n, p)
    return nx.to_directed(G)


def generate_nw_network(n, k, p):
    G = nx.newman_watts_strogatz_graph(n, k, p)
    return nx.to_directed(G)


def generate_qsn_network(n, r, q):
    G = nx.erdos_renyi_graph(n, 0.01)
    G = nx.to_directed(G)
    
    for node in list(G.nodes()):
        if random.random() < q:
            neighbors = set()
            for path_length in range(1, r + 1):
                for path in nx.all_simple_paths(G, source=node, cutoff=path_length):
                    if len(path) > 1:
                        neighbors.add(path[-1])
            
            for neighbor in neighbors:
                if neighbor != node and not G.has_edge(node, neighbor):
                    G.add_edge(node, neighbor)
    
    return G


def generate_sf_network(n, gamma=2.3):
    sequence = nx.utils.powerlaw_sequence(n, gamma)
    sequence = [int(s) for s in sequence]
    
    if sum(sequence) % 2 == 1:
        sequence[0] += 1
    
    try:
        G = nx.configuration_model(sequence)
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
    except:
        G = nx.barabasi_albert_graph(n, 3)
    
    return nx.to_directed(G)


def compute_community_division(G):
    G_undirected = G.to_undirected()
    communities = community_louvain.best_partition(G_undirected)
    
    unique_communities = sorted(set(communities.values()))
    community_map = {c: i for i, c in enumerate(unique_communities)}
    
    community_division = np.zeros(len(G), dtype=np.int64)
    for node, community in communities.items():
        community_division[node] = community_map[community]
    
    return community_division


def compute_connectivity_robustness(G, attack_strategy, batch_size=5):
    G_copy = G.copy()
    n = len(G_copy)
    batch_nodes = max(1, int(n * batch_size / 100))
    steps = n // batch_nodes
    
    robustness_seq = np.zeros(steps)
    
    for i in range(steps):
        if attack_strategy == 'ra':
            # Random attack
            nodes_to_remove = random.sample(list(G_copy.nodes()), min(batch_nodes, len(G_copy)))
        elif attack_strategy == 'tda':
            # Targeted degree attack
            degrees = dict(G_copy.degree())
            nodes_to_remove = sorted(degrees, key=degrees.get, reverse=True)[:min(batch_nodes, len(G_copy))]
        elif attack_strategy == 'tba':
            # Targeted betweenness attack
            try:
                betweenness = nx.betweenness_centrality(G_copy)
                nodes_to_remove = sorted(betweenness, key=betweenness.get, reverse=True)[:min(batch_nodes, len(G_copy))]
            except:
                degrees = dict(G_copy.degree())
                nodes_to_remove = sorted(degrees, key=degrees.get, reverse=True)[:min(batch_nodes, len(G_copy))]
        
        G_copy.remove_nodes_from(nodes_to_remove)
        
        if len(G_copy) > 0:
            largest_cc = max(nx.weakly_connected_components(G_copy), key=len)
            robustness_seq[i] = len(largest_cc) / (n - (i + 1) * batch_nodes)
        else:
            robustness_seq[i] = 0
    
    return robustness_seq


def compute_controllability_robustness(G, attack_strategy, batch_size=5):
    G_copy = G.copy()
    n = len(G_copy)
    batch_nodes = max(1, int(n * batch_size / 100))
    steps = n // batch_nodes
    
    robustness_seq = np.zeros(steps)
    
    for i in range(steps):
        if attack_strategy == 'ra':
            nodes_to_remove = random.sample(list(G_copy.nodes()), min(batch_nodes, len(G_copy)))
        elif attack_strategy == 'tda':
            degrees = dict(G_copy.degree())
            nodes_to_remove = sorted(degrees, key=degrees.get, reverse=True)[:min(batch_nodes, len(G_copy))]
        elif attack_strategy == 'tba':
            try:
                betweenness = nx.betweenness_centrality(G_copy)
                nodes_to_remove = sorted(betweenness, key=betweenness.get, reverse=True)[:min(batch_nodes, len(G_copy))]
            except:
                degrees = dict(G_copy.degree())
                nodes_to_remove = sorted(degrees, key=degrees.get, reverse=True)[:min(batch_nodes, len(G_copy))]
        
        G_copy.remove_nodes_from(nodes_to_remove)
        
        if len(G_copy) > 0:
            try:
                matching = nx.bipartite.maximum_matching(nx.DiGraph(G_copy))
                nd = max(1, n - (i + 1) * batch_nodes - len(matching) // 2)
                robustness_seq[i] = nd / (n - (i + 1) * batch_nodes)
            except:
                robustness_seq[i] = 1.0
        else:
            robustness_seq[i] = 1.0
    
    return robustness_seq


def extract_features(G):
    n = len(G)
    features = []
    
    # 1. In-degree
    in_degree = np.array([d for _, d in G.in_degree()]) / (n - 1)
    features.append(in_degree)
    
    # 2. Out-degree
    out_degree = np.array([d for _, d in G.out_degree()]) / (n - 1)
    features.append(out_degree)
    
    # 3. Clustering coefficient
    G_undirected = G.to_undirected()
    clustering = np.array(list(nx.clustering(G_undirected).values()))
    features.append(clustering)
    
    # 4. Closeness centrality
    try:
        closeness = np.array(list(nx.closeness_centrality(G).values()))
    except:
        closeness = np.zeros(n)
    features.append(closeness)
    
    # 5. PageRank
    pagerank = np.array(list(nx.pagerank(G).values()))
    features.append(pagerank)
    
    # 6-7. Hub and authority scores
    try:
        hub, authority = nx.hits(G)
        hub_scores = np.array(list(hub.values()))
        authority_scores = np.array(list(authority.values()))
    except:
        hub_scores = np.zeros(n)
        authority_scores = np.zeros(n)
    features.append(hub_scores)
    features.append(authority_scores)
    
    # Community-based features
    communities = community_louvain.best_partition(G_undirected)
    
    # 8. Community internal degree
    community_internal_degree = np.zeros(n)
    for i, node in enumerate(G.nodes()):
        node_community = communities[node]
        for neighbor in G.neighbors(node):
            if communities[neighbor] == node_community:
                community_internal_degree[i] += 1
    community_internal_degree = community_internal_degree / (n - 1)
    features.append(community_internal_degree)
    
    # 9. Community external degree
    community_external_degree = out_degree - community_internal_degree
    features.append(community_external_degree)
    
    features = np.array(features)
    return features.T


def generate_dataset(n_nodes, n_samples, output_dir, attack_id=99):
    os.makedirs(f'{output_dir}/train', exist_ok=True)
    os.makedirs(f'{output_dir}/val', exist_ok=True)
    os.makedirs(f'{output_dir}/test', exist_ok=True)
    
    network_params = {
        'ba': {'m': 3},
        'er': {'p': 0.01},
        'nw': {'k': 5, 'p': 0.01},
        'qsn': {'r': 2, 'q': 0.01},
        'sf': {'gamma': 2.3}
    }
    
    attack_strategies = ['ra', 'tda', 'tba']
    
    n_train = int(0.8 * n_samples)
    n_val = int(0.1 * n_samples)
    n_test = n_samples - n_train - n_val
    
    for method in network_params.keys():
        print(f"Generating {method.upper()} networks...")
        
        for attack in attack_strategies:
            print(f"Processing {attack.upper()} attack strategy...")
            
            train_A = []
            train_features = []
            train_labels = []
            train_community = []
            
            val_A = []
            val_features = []
            val_labels = []
            val_community = []
            
            test_A = []
            test_features = []
            test_labels = []
            test_community = []
            
            for i in tqdm(range(n_samples)):
                if method == 'ba':
                    G = generate_ba_network(n_nodes, network_params[method]['m'])
                elif method == 'er':
                    G = generate_er_network(n_nodes, network_params[method]['p'])
                elif method == 'nw':
                    G = generate_nw_network(n_nodes, network_params[method]['k'], network_params[method]['p'])
                elif method == 'qsn':
                    G = generate_qsn_network(n_nodes, network_params[method]['r'], network_params[method]['q'])
                elif method == 'sf':
                    G = generate_sf_network(n_nodes, network_params[method]['gamma'])
                
                A = nx.adjacency_matrix(G).toarray()
                features = extract_features(G)
                community_division = compute_community_division(G)
                
                connectivity_robustness = compute_connectivity_robustness(G, attack)
                controllability_robustness = compute_controllability_robustness(G, attack)
                
                label = np.vstack([connectivity_robustness, controllability_robustness])
                
                if i < n_train:
                    train_A.append(A)
                    train_features.append(features)
                    train_labels.append(label)
                    train_community.append(community_division)
                elif i < n_train + n_val:
                    val_A.append(A)
                    val_features.append(features)
                    val_labels.append(label)
                    val_community.append(community_division)
                else:
                    test_A.append(A)
                    test_features.append(features)
                    test_labels.append(label)
                    test_community.append(community_division)
            
            # Save training data
            np.save(f'{output_dir}/train/{method}_{attack}_train.npy', np.array(train_A, dtype=object))
            np.save(f'{output_dir}/train/{method}_{attack}_train_feature.npy', np.array(train_features, dtype=object))
            np.save(f'{output_dir}/train/{method}_{attack}_train_label.npy', np.array(train_labels))
            np.save(f'{output_dir}/train/{method}_{attack}_train_neighbor.npy', np.array(train_community))
            
            # Save validation data
            np.save(f'{output_dir}/val/{method}_{attack}_val.npy', np.array(val_A, dtype=object))
            np.save(f'{output_dir}/val/{method}_{attack}_val_feature.npy', np.array(val_features, dtype=object))
            np.save(f'{output_dir}/val/{method}_{attack}_val_label.npy', np.array(val_labels))
            np.save(f'{output_dir}/val/{method}_{attack}_val_neighbor.npy', np.array(val_community))
            
            # Save test data
            np.save(f'{output_dir}/test/{method}_{attack}_test.npy', np.array(test_A, dtype=object))
            np.save(f'{output_dir}/test/{method}_{attack}_test_feature.npy', np.array(test_features, dtype=object))
            np.save(f'{output_dir}/test/{method}_{attack}_test_label.npy', np.array(test_labels))
            np.save(f'{output_dir}/test/{method}_{attack}_test_neighbor.npy', np.array(test_community))
            
            np.savez(
                f'{output_dir}/test/{method}_{attack}_test.npz',
                A=test_A,
                feature=test_features,
                label=test_labels,
                neighbor=test_community
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=500, help='Number of nodes in each network (default: 500)')
    parser.add_argument('--samples', type=int, default=5000, help='Number of samples to generate (default: 5000)')
    parser.add_argument('--output_dir', type=str, default='../matrix_500nodes_5000/99', help='Output directory')
    parser.add_argument('--attack_id', type=int, default=99, help='Attack strategy ID (default: 99)')
    args = parser.parse_args()
    
    generate_dataset(args.nodes, args.samples, args.output_dir, args.attack_id)