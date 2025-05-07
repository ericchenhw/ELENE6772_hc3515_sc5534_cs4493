import time
import networkx as nx
import numpy as np
import scipy.sparse as ssp
import torch
import argparse
import os
from RP_GNN import RP_GNN

def mean_error(output, label):
    error = torch.abs(output - label)
    meanerror = error.sum() / len(output)
    return meanerror


def load_data(method, attack, test_data_path):
    test_A = []
    feature_gather = []
    label = []
    
    test_data = np.load(f'{test_data_path}/{method}_{attack}_test.npz', allow_pickle=True)
    
    for i in range(len(test_data['A'])):
        A = test_data['A'][i]
        test_A.append(A)
        label.append(test_data['label'][i])
        
        G = nx.DiGraph(A)
        degree = []
        
        # Feature extraction
        in_degree = np.array(G.in_degree)[:, 1] / (len(G) - 1)
        degree.append(in_degree)
        
        out_degree = np.array(G.out_degree)[:, 1] / (len(G) - 1)
        degree.append(out_degree)
        
        clustering = np.array(list(nx.clustering(G.to_undirected()).values()))
        degree.append(clustering)
        
        try:
            closeness = np.array(list(nx.closeness_centrality(G).values()))
        except:
            closeness = np.zeros(len(G))
        degree.append(closeness)
        
        pagerank = np.array(list(nx.pagerank(G).values()))
        degree.append(pagerank)
        
        try:
            hub, authority = nx.hits(G)
            hub_scores = np.array(list(hub.values()))
            authority_scores = np.array(list(authority.values()))
        except:
            hub_scores = np.zeros(len(G))
            authority_scores = np.zeros(len(G))
        degree.append(hub_scores)
        degree.append(authority_scores)
        
        degree = np.array(degree)
        feature_gather.append(degree.transpose())
        
    feature_gather = np.array(feature_gather)
    label = np.array(label)
    
    return feature_gather, label, test_A


def test(model, test_A, test_x, test_label, device, test_com_div):
    mean_error1 = 0
    mean_error2 = 0
    output_final = []

    for i in range(len(test_A)):
        a = test_A[i]
        x = torch.FloatTensor(test_x[i]).to(device)

        B = ssp.spmatrix.tocoo(a)
        edge_index = [B.row, B.col]
        edge_index = np.array(edge_index)
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

        model.eval()
        com_div = torch.tensor(test_com_div[i]).to(device)
        task1_output, task2_output = model(x, edge_index, com_div)

        m = torch.FloatTensor(test_label[i, :]).to(device)
        mean_error1 += mean_error(task1_output, m[0])
        mean_error2 += mean_error(task2_output, m[1])

        output_final.append([task1_output.cpu().detach().numpy(), task2_output.cpu().detach().numpy()])

    output_final = np.array(output_final)

    return output_final, mean_error1, mean_error2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='ba', help='graph method: ba, er, nw, qsn, sf')
    parser.add_argument('--attack', type=str, default='ra', help='attack strategy: ra, tda, tba')
    parser.add_argument('--attack_id', type=int, default=99, help='attack strategy ID')
    parser.add_argument('--length', type=int, default=19, help='length of the robustness sequence')
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--nodes', type=int, default=500, help='graph size')
    parser.add_argument('--com_num', type=int, default=14, help='number of communities')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

    test_data_path = f'../matrix_{args.nodes}nodes_5000/{args.attack_id}/test'
    os.makedirs(test_data_path, exist_ok=True)
    
    model = RP_GNN(args.com_num, args.length, args.nodes).to(device)
    model_path = f'../matrix_{args.nodes}nodes_5000/{args.attack_id}/{args.method}_{args.attack}_best_model.pth'
    model.load_state_dict(torch.load(model_path))

    methods = ['ba', 'er', 'nw', 'qsn', 'sf'] if args.method == 'all' else [args.method]
    attacks = ['ra', 'tda', 'tba'] if args.attack == 'all' else [args.attack]

    mean_error_connectivity = np.zeros((len(methods), len(attacks)))
    mean_error_controllability = np.zeros((len(methods), len(attacks)))
    output_final = []

    for i, method in enumerate(methods):
        for j, attack in enumerate(attacks):
            print(f"Testing {method.upper()} network with {attack.upper()} attack strategy...")
            
            begin = time.time()
            
            test_x, test_label, test_A = load_data(method, attack, test_data_path)
            test_com_div = np.load(f'{test_data_path}/{method}_{attack}_test_neighbor.npz')['array']

            output, mean_error1, mean_error2 = test(model, test_A, test_x, test_label, device, test_com_div)
            
            end = time.time()
            print(f"Test time: {end - begin:.3f}s")

            output_final.append(output)
            mean_error_connectivity[i, j] = mean_error1 / len(test_A)
            mean_error_controllability[i, j] = mean_error2 / len(test_A)

            print(f"Connectivity MAE: {mean_error_connectivity[i, j]:.5f}")
            print(f"Controllability MAE: {mean_error_controllability[i, j]:.5f}")
            
    np.savez(
        f'../matrix_{args.nodes}nodes_5000/{args.attack_id}/test_results.npz',
        connectivity=mean_error_connectivity,
        controllability=mean_error_controllability,
        output=output_final
    )