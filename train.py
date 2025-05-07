import numpy as np
import torch.optim as optim
import torch
import random
import time
from tqdm import *
import scipy.sparse as ssp
from RP_GNN import RP_GNN
from loss import MultiLoss
from rp_test import mean_error
from torch.utils.tensorboard import SummaryWriter
import argparse
import os

def val(model, test_A, test_x, test_label, device, val_com_div, args):
    model.eval()
    connectivity_val1 = 0
    controllability_val2 = 0
    output_final = []
    connectivity_mean_error = 0
    controllability_mean_error = 0
    
    for i in range(len(test_A)):
        a = test_A[i]
        x = test_x[i,:,:]
        x = x.astype(float)
        x = torch.FloatTensor(x).to(device)

        B = ssp.spmatrix.tocoo(a)
        edge_index = [B.row, B.col]
        edge_index = np.array(edge_index)
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

        com_div = torch.tensor(val_com_div[i])
        com_div = com_div.to(device)
        task1_output, task2_output = model(x, edge_index, com_div)

        m = torch.FloatTensor(test_label[i,:]).to(device)
        loss_test_task1, loss_test_task2 = MultiLoss(task1_output, task2_output, m)

        connectivity_val1 += loss_test_task1.cpu().item()
        controllability_val2 += loss_test_task2.cpu().item()
        
        controllability_mean_error += mean_error(task2_output, m[1]) / len(test_A)
        connectivity_mean_error += mean_error(task1_output, m[0]) / len(test_A)
        output_final.append(task2_output.cpu().detach().numpy())
        
    model.train()
    return connectivity_val1, controllability_val2, connectivity_mean_error, controllability_mean_error


def train(epochs, device, args):
    writer = SummaryWriter('runs/' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    
    data_dir = f'../matrix_{args.nodes}nodes_5000/{args.attack_id}'
    os.makedirs(f'{data_dir}/train', exist_ok=True)
    os.makedirs(f'{data_dir}/val', exist_ok=True)
    
    train_feature = np.load(f'{data_dir}/train/{args.method}_{args.attack}_train_feature.npy', allow_pickle=True)
    train_A = np.load(f'{data_dir}/train/{args.method}_{args.attack}_train.npy', allow_pickle=True)
    train_data_label = np.load(f'{data_dir}/train/{args.method}_{args.attack}_train_label.npy', allow_pickle=True)
    train_com_div = np.load(f'{data_dir}/train/{args.method}_{args.attack}_train_neighbor.npy', allow_pickle=True)

    val_feature = np.load(f'{data_dir}/val/{args.method}_{args.attack}_val_feature.npy', allow_pickle=True)
    val_A = np.load(f'{data_dir}/val/{args.method}_{args.attack}_val.npy', allow_pickle=True)
    val_data_label = np.load(f'{data_dir}/val/{args.method}_{args.attack}_val_label.npy', allow_pickle=True)
    val_com_div = np.load(f'{data_dir}/val/{args.method}_{args.attack}_val_neighbor.npy', allow_pickle=True)

    model = RP_GNN(args.com_num, args.length, args.nodes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    
    num_train_ins = len(train_feature)
    best_test_loss = float('inf')
    shuffle_list = random.sample(range(0, num_train_ins), num_train_ins)
    best_epoch = 0

    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}")

        loss_train_sum = 0
        begin = time.time()
        
        for i in trange(len(train_A)):
            x = train_feature[shuffle_list[i],:,:]
            x = x.astype(float)
            x = torch.FloatTensor(x)

            A = train_A[shuffle_list[i]]
            
            B = ssp.spmatrix.tocoo(A)
            edge_index = [B.row, B.col]
            edge_index = np.array(edge_index)
            edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

            com_div = torch.tensor(train_com_div[i]).to(device)
            connectivity, controllability = model(x.to(device), edge_index, com_div)
            
            label = train_data_label[shuffle_list[i],:]
            label = torch.FloatTensor(label)
            connectivity_loss, controllability_loss = MultiLoss(connectivity, controllability, label.to(device))
            
            loss_train = connectivity_loss + 2 * controllability_loss
            loss_train_sum += controllability_loss.cpu().item()

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
   
            if (i + 1) % 100 == 0:
                writer.add_scalar('Loss/connectivity', connectivity_loss, epoch * len(train_A) + i)
                writer.add_scalar('Loss/controllability', controllability_loss, epoch * len(train_A) + i)

        stop = time.time()
        print(f"Epoch time: {stop-begin:.3f}s")

        with torch.no_grad():
            connectivity_loss, controllability_loss, connectivity_mean_error, controllability_mean_error = val(
                model, val_A, val_feature, val_data_label, device, val_com_div, args
            )

        current_loss = connectivity_loss + controllability_loss
        if current_loss < best_test_loss:
            print(f"Best validation loss: {current_loss:.4f}")
            best_test_loss = current_loss
            torch.save(model.state_dict(), f'{data_dir}/{args.method}_{args.attack}_best_model.pth')
            best_epoch = epoch
            
        print(f"Connectivity mean error: {connectivity_mean_error:.4f}")
        print(f"Controllability mean error: {controllability_mean_error:.4f}")
        print(f"Epoch: {epoch + 1}, Training loss: {loss_train_sum:.4f}")
        print(f"Connectivity loss: {connectivity_loss:.4f}, Controllability loss: {controllability_loss:.4f}")
        
    writer.close()
    print(f'Best Epoch: {best_epoch + 1}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='ba', help='graph method: ba, er, nw, qsn, sf')
    parser.add_argument('--attack', type=str, default='ra', help='attack strategy: ra, tda, tba')
    parser.add_argument('--attack_id', type=int, default=99, help='attack strategy ID')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--length', type=int, default=19, help='length of the robustness sequence')
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--nodes', type=int, default=500, help='graph size')
    parser.add_argument('--com_num', type=int, default=14, help='number of communities')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    train(args.epochs, device, args)