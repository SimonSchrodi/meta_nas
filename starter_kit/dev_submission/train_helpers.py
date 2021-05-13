"""
modified copy of torch_evaluator
"""

import numpy as np
import random
import time
import warnings


import torch
import torch.nn as nn
import torch.optim as optim


import torch
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


# === TOP-K ACCURACY CALCULATION =======================================================================================
def top_k_accuracy(output, target, top_k):
    if len(output.shape) == 2:
        output = output.reshape(list(output.shape) + [1, 1])
        target = target.reshape(list(target.shape) + [1, 1])
    correct = np.zeros(len(top_k))
    _, pred = output.topk(max(top_k), 1, True, True)
    for i, k in enumerate(top_k):
        target_expand = target.unsqueeze(1).repeat(1, k, 1, 1)
        equal = torch.max(pred[:, :k, :, :].eq(target_expand), 1)[0]
        correct[i] = torch.sum(equal)
    return correct, len(target.view(-1)), equal.cpu().numpy()


# === MODEL UTILITIES ==================================================================================================
def general_num_params(m):
    # return number of differential parameters of input model
    return sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, m.parameters())])


def reset_weights(model):
    warn_non_resets = []
    diff_non_resets = []
    for module in model.modules():
        if type(module) != type(model):
            if 'reset_parameters' in dir(module):
                module.reset_parameters()
            else:
                if 'parameters' in dir(module):
                    n_params = general_num_params(module)
                    child_params = sum([general_num_params(m) for m in module.children()])

                    if n_params != 0 and n_params != child_params:
                        diff_non_resets.append([type(module).__name__, n_params])
                else:
                    warn_non_resets.append(type(module).__name__)

    if len(diff_non_resets):
        error_string = "\n".join(["\t* {}: {:,} parameter(s)".format(m, p) for m, p in diff_non_resets])
        raise AttributeError(
            "{} module(s) have differentiable parameters without a 'reset_parameters' function: \n {}".format(
                len(diff_non_resets),
                error_string))
    if len(warn_non_resets):
        warning_msg = "Model contains modules without 'reset_parameters' attribute: "
        warnings.warn(warning_msg + str(set(warn_non_resets)))

# === TRAIN/TEST FUNCTIONS =============================================================================================
def train(model, device, optimizer, criterion, train_loader):
    # === train epoch =========================
    model.train()

    for i, (data, target) in enumerate(train_loader):
        # pass data ===========================
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model.forward(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def evaluate(model, device, criterion, loader):
    # === tracking stats =====================
    corrects, divisor = 0, 0

    # === test epoch =========================
    model.eval()
    with torch.no_grad():
        for _, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            output = model.forward(data)
            _ = criterion(output, target)

            corr, div, _ = top_k_accuracy(output, target, top_k=[1])
            corrects += corr
            divisor += div
            acc = 100 * corrects / float(divisor)
    return acc.item()


# === FULL N EPOCH TRAIN =============================================================================================
def full_training(model, **kwargs):
    # grab necessary kwargs
    device = kwargs['device']
    epochs = kwargs['epochs']
    lr = kwargs['lr']
    train_loader = kwargs['train_loader']
    valid_loader = kwargs['valid_loader']

    inc_time_limit = kwargs['inc_time_limit']
    search_time_limit = kwargs['search_time_limit']

    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=.9, weight_decay=3e-4)

    # sum reduction to match tensorflow
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0
    train_start = time.time()
    for epoch in range(epochs):
        train(model, device, optimizer, criterion, train_loader)
        valid_acc = evaluate(model, device, criterion, valid_loader)
        print('finished epoch', epoch)

        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
        scheduler.step()

        average_epoch_t = (time.time() - train_start) / (epoch + 1)
        estimated_train_time = average_epoch_t * epochs
        if time.time() > inc_time_limit or (train_start + estimated_train_time > search_time_limit and epoch >= 2):
            return best_val_acc, estimated_train_time

    return best_val_acc, time.time() - train_start