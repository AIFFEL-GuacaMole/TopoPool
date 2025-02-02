import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm

from util import load_data, separate_data
from models.witness_graphcnn import GraphCNN


criterion = nn.CrossEntropyLoss()

def train(args, model, device, train_graphs, optimizer, epoch):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output = model(batch_graph)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        #compute loss
        loss = criterion(output, labels)

        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        #report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum/total_iters
    print("loss training: %f" % (average_loss))
    
    # Log training loss to WandB
    wandb.log({"train_loss": average_loss}, step=epoch)
    
    return average_loss

def pass_data_iteratively(model, graphs, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

def test(args, model, device, train_graphs, test_graphs, epoch):
    model.eval()

    output = pass_data_iteratively(model, train_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    print("accuracy train: %f test: %f" % (acc_train, acc_test))

    # Log accuracies to WandB
    wandb.log({
        "train_accuracy": acc_train,
        "test_accuracy": acc_test,
    }, step=epoch)

    return acc_train, acc_test

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="PTC",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=350,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=9,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=8,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.0,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true",
                        help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type=str, default="",
                        help='output file')
    parser.add_argument('--wandb_project', type=str, default="Wit-TopoPool",
                        help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='WandB entity name')
    args = parser.parse_args()

    # Initialize WandB
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "dataset": args.dataset,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "num_layers": args.num_layers,
            "hidden_dim": args.hidden_dim,
            "graph_pooling": args.graph_pooling_type,
            "neighbor_pooling": args.neighbor_pooling_type,
            "learn_eps": args.learn_eps,
            "degree_as_tag": args.degree_as_tag
        }
    )

    #set up seeds and gpu device
    random_seed = 0
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = (
        torch.device(f"cuda:{str(args.device)}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    graphs, num_classes = load_data(args.dataset, args.degree_as_tag)

    train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)

    model = GraphCNN(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, device).to(device)

    # Log model architecture
    wandb.watch(model, log="all")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    max_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        scheduler.step()

        # Log learning rate
        wandb.log({"learning_rate": optimizer.param_groups[0]['lr']}, step=epoch)

        avg_loss = train(args, model, device, train_graphs, optimizer, epoch)
        acc_train, acc_test = test(args, model, device, train_graphs, test_graphs, epoch)

        max_acc = max(max_acc, acc_test)
        print("Current best result is:", max_acc)

        if args.filename != "":
            with open(args.filename, 'a+') as f:
                f.write("%f %f %f" % (avg_loss, acc_train, acc_test))
                f.write("\n")
        print("")

        # Log epsilon values if learn_eps is True
        if args.learn_eps:
            wandb.log({"epsilon": model.eps.item()}, step=epoch)
            print(model.eps)

    # Log final best accuracy
    wandb.log({"best_test_accuracy": max_acc})

    with open(f'{str(args.dataset)}acc_results.txt', 'a+') as f:
        f.write(str(max_acc) + '\n')

    # Close WandB run
    wandb.finish()

if __name__ == '__main__':
    main()