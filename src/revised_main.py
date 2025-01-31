from rdkit import Chem
from sklearn.metrics import average_precision_score
from tdc.benchmark_group import admet_group
from tdc.single_pred import ADME
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import networkx as nx
import numpy as np
import torch
import wandb

from models.witness_graphcnn import GraphCNN
from revised_util import mol_to_s2v_graph, prepare_molecular_dataset


# Define the criterion based on the task type later in the code
# For now, initialize it as binary; will adjust in main()
criterion = nn.BCEWithLogitsLoss()

benchmark_config = {
    'cyp1a2_veith': ('binary', False),
    'cyp2c19_veith': ('binary', False),
    'cyp2c9_veith': ('binary', False),
    'cyp2d6_veith': ('binary', False),
    'cyp3a4_veith': ('binary', False),
    'caco2_wang': ('regression', False),

    'bioavailability_ma': ('binary', False),
    'lipophilicity_astrazeneca': ('regression', False),
    'solubility_aqsoldb': ('regression', False),
    'hia_hou': ('binary', False),
    'pgp_broccatelli': ('binary', False),
    'bbb_martins': ('binary', False),
    'ppbr_az': ('regression', False),
    'vdss_lombardo': ('regression', True),

    'cyp2c9_substrate_carbonmangels': ('binary', False),
    'cyp2d6_substrate_carbonmangels': ('binary', False),
    'cyp3a4_substrate_carbonmangels': ('binary', False),
    'half_life_obach': ('regression', True),
    'clearance_hepatocyte_az': ('regression', True),
    'clearance_microsome_az': ('regression', True),
    'ld50_zhu': ('regression', False),
    'herg': ('binary', False),
    'ames': ('binary', False),
    'dili': ('binary', False)
}

# Custom collate function to handle batches of S2VGraph objects
def collate_fn(batch):
    """
    Custom collate function to handle batches of S2VGraph objects.
    :param batch: List of S2VGraph objects
    :return: List of S2VGraph objects
    """
    return batch

class GraphDataset(Dataset):
    def __init__(self, graphs):
        """
        Initializes the dataset with a list of graph objects.
        :param graphs: List of S2VGraph objects.
        """
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

def train(args, model, device, train_graphs, optimizer, epoch, criterion):
    model.train()

    train_dataset = GraphDataset(train_graphs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=collate_fn
    )

    loss_accum = 0
    pbar = tqdm(train_loader, unit='batch')

    for valid_batches, batch_graph in enumerate(pbar, start=1):
        batch_graph = list(batch_graph)  # Ensure it's a list
        output = model(batch_graph)
        labels = torch.FloatTensor([graph.label for graph in batch_graph]).to(device)
        output = output.squeeze(1)
        
        # Compute loss
        loss = criterion(output, labels)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        # Accumulate loss
        loss_accum += loss.detach().cpu().item()
        # Update progress bar with valid loss
        if valid_batches > 0:
            avg_loss = loss_accum / valid_batches
            pbar.set_description(f'Epoch: {epoch} - Loss: {avg_loss:.4f}')

    average_loss = loss_accum / len(train_loader)
    print(f"Average training loss: {average_loss:.6f}")
    # wandb.log({"Average training loss": average_loss})
    
    return average_loss

def pass_data_iteratively(model, graphs, minibatch_size=64):
    """
    Pass data to the model in minibatches to avoid memory overflow.
    :param model: Trained model
    :param graphs: List of S2VGraph objects
    :param minibatch_size: Size of each minibatch
    :return: Concatenated output tensor
    """
    model.eval()
    output = []
    dataset = GraphDataset(graphs)
    loader = DataLoader(
        dataset,
        batch_size=minibatch_size,
        shuffle=False,
        num_workers=1,  # Adjust based on your system
        collate_fn=collate_fn
    )

    with torch.no_grad():
        for batch_graph in loader:
            batch_graph = list(batch_graph)
            batch_output = model(batch_graph)
            output.append(batch_output)

    return torch.cat(output, 0)

def test(args, model, device, train_graphs, test_graphs, epoch, criterion, task_type):
    model.eval()

    def process_outputs(output, labels):
        # Remove any NaN values
        valid_mask = ~torch.isnan(output)
        output = output[valid_mask]
        labels = labels[valid_mask]
        
        # Ensure we have valid data
        if len(output) == 0:
            print("Warning: No valid predictions after filtering NaN values")
            return None
            
        # Convert to numpy and ensure correct shape
        output = output.cpu().detach().numpy()
        labels = labels.cpu().numpy()
        
        return output, labels

    # Evaluate on Training Data
    with torch.no_grad():
        train_output = pass_data_iteratively(model, train_graphs)
        train_labels = torch.tensor([graph.label for graph in train_graphs], 
                                  device=device, dtype=torch.float)
        
        # Process outputs
        processed_data = process_outputs(train_output.squeeze(1), train_labels)
        if processed_data is None:
            print("Warning: Could not compute training metrics due to invalid outputs")
            return 0.0, 0.0
            
        train_output, train_labels = processed_data
    
    # Evaluate on Testing Data
    test_output = pass_data_iteratively(model, test_graphs).cpu().detach().numpy()
    test_labels = np.array([graph.label for graph in test_graphs])
    test_output = test_output.squeeze(1)
    
    if task_type == 'binary':
        train_ap = average_precision_score(train_labels, train_output, average="macro")
        test_ap = average_precision_score(test_labels, test_output, average="macro")
    elif task_type == 'regression':
        # Replace with an appropriate regression metric, e.g., R2 score
        test_ap = None  # Placeholder
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    print(f"Epoch: {epoch} - Train AP: {train_ap:.4f}, Test AP: {test_ap:.4f}")
    # wandb.log({"Train AP": train_ap, "Test AP": test_ap})
    
    return train_ap, test_ap

def main(args=None):
    # If no args provided, use default args from config_loader
    if args is None:
        from config_loader import args
    
    # Set up seeds and GPU device
    random_seed = args.seed
    num_classes = 1  # Adjust based on your task

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = (
        torch.device(f"cuda:{str(args.device)}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("device:", device)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # Access a specific benchmark configuration
    benchmark_name = list(benchmark_config.keys())[2]  # e.g., 'cyp2c9_veith'
    group = admet_group(path='data/')
    benchmark = group.get(benchmark_name)

    # Prepare datasets
    train_graphs, test_graphs = prepare_molecular_dataset(benchmark['train_val'], benchmark['test'])

    # **Final Verification**
    num_none_train_final = sum(1 for g in train_graphs if g is None)
    num_none_test_final = sum(1 for g in test_graphs if g is None)
    print(f"After preparation, `None` graphs in training set: {num_none_train_final}")
    print(f"After preparation, `None` graphs in testing set: {num_none_test_final}")

    # Ensure no `None` graphs remain
    assert len(train_graphs) > 0, "No valid training graphs available after preparation."
    assert len(test_graphs) > 0, "No valid testing graphs available after preparation."
    assert num_none_train_final == 0, "There are `None` graphs in the training set."
    assert num_none_test_final == 0, "There are `None` graphs in the testing set."


    # Determine task type based on benchmark_config
    task_type, _ = benchmark_config[benchmark_name]

    # Adjust criterion based on task type
    if task_type == 'binary':
        criterion = nn.BCEWithLogitsLoss()
    elif task_type == 'regression':
        criterion = nn.MSELoss()  # Or another appropriate regression loss
        num_classes = 1  # For regression, typically one output
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # Initialize the model with `pi_dimension`
    model = GraphCNN(
        device=device,
        num_layers=args.num_layers,
        num_mlp_layers=args.num_mlp_layers,
        input_dim=train_graphs[0].node_features.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=num_classes,
        final_dropout=args.final_dropout,
        learn_eps=args.learn_eps,
        graph_pooling_type=args.graph_pooling_type,
        neighbor_pooling_type=args.neighbor_pooling_type,
        num_neighbors=args.num_neighbors,
        num_landmarks=args.num_landmarks,
        first_pool_ratio=args.first_pool_ratio,
        PI_resolution_sq=args.PI_resolution_sq,
        PI_hidden_dim=args.PI_hidden_dim,
    ).to(device)

    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    
    wandb.init(
        project="TopoPool-2",
        config=dict({"input_dim": train_graphs[0].node_features.shape[1],
                     "output_dim": num_classes,
                    },
                    **vars(args))
    )
    
    max_ap = 0.0
    for epoch in range(1, args.epochs + 1):
        scheduler.step()

        avg_loss = train(args, model, device, train_graphs, optimizer, epoch, criterion)
        ap_train, ap_test = test(args, model, device, train_graphs, test_graphs, epoch, criterion, task_type)

        # Log metrics to W&B
        wandb.log({
            "epoch": epoch,
            "Training Loss (avg)": avg_loss,
            "Train AP": ap_train,
            "Test AP": ap_test,
            "Learning rate": scheduler.get_last_lr()[0]
        }) 
        
        if task_type == 'binary':
            if ap_test > max_ap:
                max_ap = ap_test
                best_epoch = epoch
            print(f"Current best result is: {max_ap} (epoch:{best_epoch})")
        # elif task_type == 'regression':
        #     # Handle regression metrics if applicable
        #     # Example: Track best R2 score or lowest MSE
        #     pass  # Replace with appropriate logic

        if args.filename:
            with open(args.filename, 'a+') as f:
                f.write(f"{avg_loss} {ap_train} {ap_test}\n")
        print("")

        # Ensure that 'eps' attribute exists and is printed correctly
        if hasattr(model, 'eps'):
            print(model.eps)
        else:
            print("Model does not have 'eps' attribute.")

    # Save the best AP score
    # with open(f"{args.dataset}_ap_results.txt", 'a+') as f:
    with open("cyp2c9_veith--AP_results.txt", 'a+') as f:
        f.write(f"{max_ap}\n")
    
    # Log final metrics
    wandb.log({
        "best test AP": max_ap,
        "best score epoch": best_epoch
    })

    return max_ap  # Return best performance for sweep optimization

if __name__ == '__main__':
    main()