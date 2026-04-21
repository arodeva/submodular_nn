import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
import os
import sys
import wandb
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dqn_old import IncreasingConcaveNet, concavity_regularizer, MonotoneSubmodularNet

def log_fn(S):
    total = sum(torch.sum(z) for z in S)
    return torch.log(total + 1e-6)

def logdet_fn(S):
    d = S[0].shape[0]
    A = torch.eye(d)
    for z in S:
        A += torch.ger(z, z)
    return torch.logdet(A + 1e-6 * torch.eye(d))

def facility_location_fn(S, V):
    total = 0
    for z0 in V:
        sims = [torch.dot(z, z0) / (torch.norm(z) * torch.norm(z0) + 1e-6) for z in S]
        total += max(sims)
    return total

def monotone_graph_cut_fn(S, V, sigma=0.01):
    cut = sum(torch.dot(u, v) for u in V for v in S)
    internal = sum(torch.dot(u, v) for u in S for v in S)
    return cut - sigma * internal

def non_monotone_graph_cut_fn(S, V):
    return monotone_graph_cut_fn(S, V, sigma=0.8)

class SubmodularSetDataset(Dataset):
    def __init__(self, V, function_name, precomputed_data=None, precomputed_labels=None, use_binary=True, seed=None):
        self.V = V
        self.d = V.shape[1]
        self.N = V.shape[0]
        self.use_binary = use_binary
        self.function_name = function_name
        funcs = {
            "log": log_fn,
            "logdet": logdet_fn,
            "fl": facility_location_fn,
            "monotone_gcut": monotone_graph_cut_fn,
            "non_monotone_gcut": non_monotone_graph_cut_fn,
        }
        self.f = funcs[function_name]

        if precomputed_data is not None and precomputed_labels is not None:
            self.data = precomputed_data
            self.labels = precomputed_labels
            self.n_subsets = len(self.data)
            return

        if seed is not None:
            torch.manual_seed(seed)

        data_list = []
        labels_list = []
        perm = torch.randperm(self.N).tolist()
        for k in range(1, self.N + 1):
            indices = perm[:k]
            S = [self.V[i] for i in indices]
            if self.use_binary:
                x = torch.zeros(self.N, dtype=torch.float)
                if len(indices) > 0:
                    x[torch.tensor(indices, dtype=torch.long)] = 1.0
            else:
                x = torch.stack(S).sum(dim=0) if len(S) > 0 else torch.zeros(self.d)
            y = self.f(S) if function_name != "fl" and ("gcut" not in function_name) else self.f(S, self.V)
            data_list.append(x)
            labels_list.append(y)

        self.data = torch.stack(data_list)
        self.labels = torch.tensor(labels_list).float()
        self.n_subsets = self.N

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'V': self.V,
            'function_name': self.function_name,
            'use_binary': self.use_binary,
            'data': self.data,
            'labels': self.labels,
        }, path)

    @classmethod
    def load_from_file(cls, path):
        d = torch.load(path)
        return cls(d['V'], d['function_name'], precomputed_data=d.get('data'),
                   precomputed_labels=d.get('labels'), use_binary=d.get('use_binary', True))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train(function_name="log", learning_rate=1e-3, dataset_path=None, regenerate_dataset=False, seed=None, trial_idx=0):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    n = int(1e4)
    d = 10
    V = torch.rand(n, d)

    if dataset_path is not None and os.path.exists(dataset_path) and not regenerate_dataset:
        print(f"Loading dataset from {dataset_path}")
        dataset = SubmodularSetDataset.load_from_file(dataset_path)
    else:
        print("Generating dataset...")
        dataset = SubmodularSetDataset(V, function_name, use_binary=True)
        if dataset_path is not None:
            print(f"Saving dataset to {dataset_path}")
            dataset.save(dataset_path)

    batch_size = 32
    total_len = len(dataset)
    base = total_len // 3
    lengths = [base, base, total_len - 2 * base]
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, lengths)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    phi_layers = [1, 100, 100, 1]
    lamb = 0.5
    m_layers = 2
    m_size = 1  # old style: each element processed as a scalar via unsqueeze(-1)

    model = MonotoneSubmodularNet(phi_layers, lamb, m_layers, m_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    num_epochs = 300
    best_val_rmse = float('inf')
    best_model_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_mse_sum = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            running_mse_sum += loss.item() * batch_X.size(0)
            model.clamp_weights(hard_enforce=True)

        epoch_rmse = math.sqrt(running_mse_sum / len(train_dataset))

        model.eval()
        val_mse_sum = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                mse_loss = criterion(outputs.squeeze(), batch_y)
                val_mse_sum += mse_loss.item() * batch_X.size(0)
        val_rmse = math.sqrt(val_mse_sum / len(val_dataset))

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch}/{num_epochs} | Train RMSE: {epoch_rmse:.4f}, Val RMSE: {val_rmse:.4f}")
        wandb.log({
            "epoch": epoch,
            f"trial_{trial_idx}/Train RMSE": epoch_rmse,
            f"trial_{trial_idx}/Val RMSE": val_rmse,
        })

    model.load_state_dict(best_model_state)
    print(f"Loaded best model (val RMSE: {best_val_rmse:.4f})")

    model.eval()
    test_mse_sum = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            mse_loss = criterion(outputs.squeeze(), batch_y)
            test_mse_sum += mse_loss.item() * batch_X.size(0)
    test_rmse = math.sqrt(test_mse_sum / len(test_dataset))
    print(f"Final Test RMSE: {test_rmse:.4f}")
    wandb.log({f"trial_{trial_idx}/Test RMSE": test_rmse})
    return test_rmse


def run_trials(function_name, n_trials=5, learning_rate=1e-3, regenerate_dataset=False):
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "cached_datasets", f"{function_name}_old_binary.pt")
    seeds = [42 + i for i in range(n_trials)]

    wandb.init(project="submodular_nn", config={
        "function": function_name,
        "n_trials": n_trials,
        "learning_rate": learning_rate,
        "use_binary": True,
        "style": "old",
    })

    results = []
    for i, seed in enumerate(seeds):
        print(f"\n=== Trial {i+1}/{n_trials} (seed={seed}) ===")
        rmse = train(
            function_name=function_name,
            learning_rate=learning_rate,
            dataset_path=dataset_path,
            regenerate_dataset=(regenerate_dataset and i == 0),
            seed=seed,
            trial_idx=i,
        )
        results.append(rmse)
        print(f"Trial {i+1} Test RMSE: {rmse:.4f}")

    mean = np.mean(results)
    std = np.std(results)
    print(f"\n=== Results over {n_trials} trials ===")
    print(f"Test RMSE: {mean:.4f} ± {std:.4f}")
    wandb.summary["Mean Test RMSE"] = mean
    wandb.summary["Std Test RMSE"] = std
    wandb.finish()
    return mean, std


if __name__ == "__main__":
    function_name = "log"
    run_trials(function_name=function_name, n_trials=5, learning_rate=1e-3, regenerate_dataset=False)
