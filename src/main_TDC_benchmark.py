from rdkit import Chem
from sklearn.metrics import average_precision_score, mean_squared_error
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
import pandas as pd

# from models.witness_graphcnn import GraphCNN
from models.gnn_realization import GCNconv as GraphCNN
from revised_util import mol_to_s2v_graph, prepare_molecular_dataset


# Define the criterion based on the task type later in the code
# For now, initialize it as binary; will adjust in main()
criterion = nn.BCEWithLogitsLoss()

benchmark_config = {
    'cyp1a2_veith': ('binary', False),  #0
    'cyp2c19_veith': ('binary', False), #1
    'cyp2c9_veith': ('binary', False),  #2
    'cyp2d6_veith': ('binary', False),  #3
    'cyp3a4_veith': ('binary', False),  #4
    'caco2_wang': ('regression', False),#5

    'bioavailability_ma': ('binary', False),             #6
    'lipophilicity_astrazeneca': ('regression', False),  #7
    'solubility_aqsoldb': ('regression', False),         #8
    'hia_hou': ('binary', False),                        #9
    'pgp_broccatelli': ('binary', False),                #10
    'bbb_martins': ('binary', False),                    #11
    'ppbr_az': ('regression', False),                    #12
    'vdss_lombardo': ('regression', True),               #13

    'cyp2c9_substrate_carbonmangels': ('binary', False), #14
    'cyp2d6_substrate_carbonmangels': ('binary', False), #15
    'cyp3a4_substrate_carbonmangels': ('binary', False), #16
    'half_life_obach': ('regression', True),             #17
    'clearance_hepatocyte_az': ('regression', True),     #18
    'clearance_microsome_az': ('regression', True),      #19
    'ld50_zhu': ('regression', False),                   #20
    'herg': ('binary', False),                           #21
    'ames': ('binary', False),                           #22
    'dili': ('binary', False)                            #23
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
    wandb.log({"Training Loss (avg)": average_loss, "epoch": epoch})
    
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
    
    # Training set evaluation
    train_loader = DataLoader(GraphDataset(train_graphs), batch_size=args.batch_size, 
                            shuffle=False, num_workers=1, collate_fn=collate_fn)
    y_true_train = []
    y_pred_train = []
    
    with torch.no_grad():
        for _, batch_graph in enumerate(train_loader):
            pred = model(batch_graph)
            labels = torch.FloatTensor([graph.label for graph in batch_graph]).to(device)
            y_true_train.extend(labels.cpu().numpy().tolist())
            
            if task_type == 'binary':
                pred = torch.sigmoid(pred)
            y_pred_train.extend(pred.cpu().numpy().flatten().tolist())
    
    # Test set evaluation
    test_loader = DataLoader(GraphDataset(test_graphs), batch_size=args.batch_size, 
                           shuffle=False, num_workers=1, collate_fn=collate_fn)
    y_true_test = []
    y_pred_test = []
    
    with torch.no_grad():
        for _, batch_graph in enumerate(test_loader):
            pred = model(batch_graph)
            labels = torch.FloatTensor([graph.label for graph in batch_graph]).to(device)
            y_true_test.extend(labels.cpu().numpy().tolist())
            
            if task_type == 'binary':
                pred = torch.sigmoid(pred)
            y_pred_test.extend(pred.cpu().numpy().flatten().tolist())
    
    # Calculate metrics based on task type
    if task_type == 'binary':
        train_metric = average_precision_score(y_true_train, y_pred_train)
        test_metric = average_precision_score(y_true_test, y_pred_test)
        print(f'Epoch: {epoch} - Train AP: {train_metric:.4f}, Test AP: {test_metric:.4f}')
    else:  # regression
        train_metric = mean_squared_error(y_true_train, y_pred_train)
        test_metric = mean_squared_error(y_true_test, y_pred_test)
        print(f'Epoch: {epoch} - Train MSE: {train_metric:.4f}, Test MSE: {test_metric:.4f}')
    
    return train_metric, test_metric

def main(args=None):
    # If no args provided, use default args from config_loader
    if args is None:
        from config_loader import args
    
    # Set up device
    device = (
        torch.device(f"cuda:{str(args.device)}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("device:", device)

    # Initialize group and predictions list for TDC evaluation
    group = admet_group(path='data/')
    predictions_list = []
    
    # Get benchmark name
    benchmark_name = list(benchmark_config.keys())[2]  # You can modify this to test different benchmarks
    print(f"Selected benchmark: {benchmark_name}")
    
    # Run evaluation with multiple seeds as per TDC guidelines
    for seed in [1, 2, 3, 4, 5]:
        print(f"\nRunning evaluation with seed {seed}")
        
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        # Get benchmark data with current seed
        benchmark = group.get(benchmark_name)
        train_val, test_data = benchmark['train_val'], benchmark['test']
        train_data, valid = group.get_train_valid_split(benchmark=benchmark_name, split_type='default', seed=seed)
        
        print(f"\nDebug Info for seed {seed} - Initial Data:")
        print(f"Raw test data size: {len(test_data)}")
        
        # Prepare datasets
        train_graphs = prepare_molecular_dataset(train_data, None)[0]
        valid_graphs = prepare_molecular_dataset(valid, None)[0]
        
        # Prepare test data with tracking of skipped molecules
        test_graphs = []
        skipped_indices = []
        total_molecules = len(test_data)
        
        print(f"\nProcessing test data for seed {seed}...")
        for idx, row in test_data.iterrows():
            try:
                single_mol_df = pd.DataFrame([row])
                processed_graphs = prepare_molecular_dataset(single_mol_df, None)[0]
                
                if processed_graphs and len(processed_graphs) > 0 and processed_graphs[0] is not None:
                    test_graphs.append(processed_graphs[0])
                else:
                    skipped_indices.append(idx)
            except Exception as e:
                skipped_indices.append(idx)
        
        processed_count = len(test_graphs)
        print(f"Processed {processed_count}/{total_molecules} molecules" + 
              (f" (skipped {len(skipped_indices)})" if skipped_indices else ""))
        
        if len(test_graphs) == 0:
            print("No valid test graphs were created - skipping this seed")
            continue
        
        # Remove None values from graphs (molecules that couldn't be processed)
        train_graphs = [g for g in train_graphs if g is not None]
        valid_graphs = [g for g in valid_graphs if g is not None]
        
        print(f"Test data size after preparation: {len(test_graphs)}")
        print(f"Number of None values removed from test data: {len(test_data) - len(test_graphs)}")
        
        print("Dataset sizes:")
        print(f"Train: {len(train_data)}")
        print(f"Valid: {len(valid)}")
        print(f"Test: {len(test_data)}")
        
        # Verify datasets
        for dataset, name in [(train_graphs, 'training'), (valid_graphs, 'validation'), (test_graphs, 'test')]:
            print(f"{name.capitalize()} set size after preparation: {len(dataset)}")
            assert len(dataset) > 0, f"No valid {name} graphs available after preparation."

        # Determine task type and set up criterion
        task_type, _ = benchmark_config[benchmark_name]
        if task_type == 'binary':
            criterion = nn.BCEWithLogitsLoss()
            num_classes = 1
        elif task_type == 'regression':
            criterion = nn.MSELoss()
            num_classes = 1
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        # Initialize model
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

        # Initialize wandb for this seed
        run_name = f"{benchmark_name}_seed_{seed}"
        wandb.init(
            project="TopoPool-2",
            name=run_name,
            config=dict({
                "benchmark_name": benchmark_name,
                "seed": seed,
                "input_dim": train_graphs[0].node_features.shape[1],
                "output_dim": num_classes,
            }, **vars(args)),
            reinit=True
        )

        # Training loop
        best_valid_metric = float('-inf') if task_type == 'binary' else float('inf')
        best_epoch = 0
        best_model_state = None
        patience = args.patience if hasattr(args, 'patience') else 50  # Default patience value
        patience_counter = 0
        
        for epoch in range(1, args.epochs + 1):
            scheduler.step()

            # Train
            avg_loss = train(args, model, device, train_graphs, optimizer, epoch, criterion)
            
            # Evaluate
            train_metric, valid_metric = test(args, model, device, train_graphs, valid_graphs, epoch, criterion, task_type)
            
            # Print train and test AP
            # print(f"Epoch: {epoch} - Train AP: {train_metric:.4f}, Test AP: {valid_metric:.4f}")
            print(f"Current best result is: {best_valid_metric} (epoch:{best_epoch})")
            print("")  # Add line break between epochs

            # Early stopping check
            improved = (task_type == 'binary' and valid_metric > best_valid_metric) or \
                      (task_type == 'regression' and valid_metric < best_valid_metric)
            
            if improved:
                best_valid_metric = valid_metric
                best_epoch = epoch
                best_model_state = model.state_dict().copy()
                patience_counter = 0

        # Load best model for final evaluation
        model.load_state_dict(best_model_state)
        
        # Get test predictions
        model.eval()
        test_loader = DataLoader(GraphDataset(test_graphs), batch_size=args.batch_size, 
                               shuffle=False, num_workers=1, collate_fn=collate_fn)
        y_pred_test = []
        
        print(f"\nDebug Info for seed {seed}:")
        print(f"Number of test graphs: {len(test_graphs)}")
        
        with torch.no_grad():
            for batch_idx, batch_graph in enumerate(test_loader):
                pred = model(batch_graph)
                if task_type == 'binary':
                    pred = torch.sigmoid(pred)
                batch_preds = pred.cpu().numpy().flatten().tolist()
                y_pred_test.extend(batch_preds)
                print(f"Batch {batch_idx}: Added {len(batch_preds)} predictions. Total predictions: {len(y_pred_test)}")
        
        print(f"Final number of predictions for seed {seed}: {len(y_pred_test)}")
        
        # Insert None or default value for skipped molecules
        full_predictions = []
        pred_idx = 0
        for i in range(len(test_data)):
            if i in skipped_indices:
                full_predictions.append(0.0)  # Use an appropriate default value
            else:
                full_predictions.append(y_pred_test[pred_idx])
                pred_idx += 1
        
        # Store predictions for this seed
        predictions = {benchmark_name: full_predictions}
        predictions_list.append(predictions)
        
        # Clean up
        wandb.finish()

    # Debug print before evaluation
    print("\nFinal Debug Info:")
    print(f"Original test data size: {len(test_data)}")
    for idx, pred_dict in enumerate(predictions_list):
        print(f"Seed {idx + 1} predictions length: {len(pred_dict[benchmark_name])}")
    
    # Additional check for TDC test labels
    test_labels = test_data['Y'].values
    print(f"TDC test labels length: {len(test_labels)}")
    
    # Debug the first prediction dictionary
    if predictions_list:
        first_pred = predictions_list[0]
        print("\nDetailed prediction info for first seed:")
        print(f"Prediction dictionary keys: {first_pred.keys()}")
        print(f"Benchmark name: {benchmark_name}")
        print(f"First few predictions: {first_pred[benchmark_name][:5]}")
        print(f"First few test labels: {test_labels[:5]}")
    
    # Evaluate results across all seeds using TDC
    try:
        results = group.evaluate_many(predictions_list)
    except Exception as e:
        print("\nError during evaluation:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        raise
    
    # Evaluate results across all seeds using TDC
    print("\nFinal TDC Evaluation Results:")
    print(f"Benchmark: {benchmark_name}")
    print(f"Average Performance: {results[benchmark_name][0]:.4f}")
    print(f"Standard Deviation: {results[benchmark_name][1]:.4f}")

    # Save predictions for leaderboard submission
    import json
    submission = {
        'model_name': 'TopoPool',
        'benchmark_group': 'ADMET_Group',
        'benchmark_name': benchmark_name,
        'predictions': predictions_list
    }
    
    with open(f'tdc_submission_{benchmark_name}.json', 'w') as f:
        json.dump(submission, f)
    
    print(f"\nPredictions saved to tdc_submission_{benchmark_name}.json")
    print("You can now submit these results to the TDC leaderboard!")

if __name__ == '__main__':
    main()