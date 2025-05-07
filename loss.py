import torch

def MultiLoss(connectivity, controllability, label):
    connectivity_loss = weighted_mseloss(connectivity, label[0])
    controllability_loss = weighted_mseloss(controllability, label[1])

    return connectivity_loss, controllability_loss


def weighted_mseloss(input, target):
    target = target.squeeze()
    a = torch.ones(input.shape).to(input.device)
    loss_vector = (input - target) ** 2
    
    # Give double weight to first half of sequence
    m = int((a.shape[0] + 1) / 2)
    a[0:m] = 2

    return torch.dot(a, loss_vector) / a.shape[0]