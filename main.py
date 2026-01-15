import argparse
import csv
import json
import os
import time
import pandas as pd
import torch
import torch.nn.functional as F
from data_loader import load_data, GeoCGNN_data_utils
from torch.utils.data import DataLoader
from model.SDCGNN import SDCGNN, MEGNET_Conv
import numpy as np
import random
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

CUDA_LAUNCH_BLOCKING = 1

# Ensure mid_out directory exists (create if not exists)
os.makedirs('mid_out', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=666, help='random_seed')
parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
parser.add_argument('--nhid', type=int, default=128, help='dimension')
parser.add_argument('--lr', type=float, default=0.001, help='learning_rate')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
parser.add_argument('--device', type=str, default='cuda', help='training devices')
parser.add_argument('--dataset', type=str, default='data', help='the path of dataset')
parser.add_argument('--neighbors', type=int, default=16, help='the max number of neighbors')
parser.add_argument('--cat_or_add', type=str, default="cat")
parser.add_argument('--a', type=float, default=0.5, help='weight of G_cs')
parser.add_argument('--b', type=float, default=0.5, help='weight of G_in')
parser.add_argument("--n_MLP_LR", type=int, default=2)
parser.add_argument("--n_grid_K", type=int, default=4)
parser.add_argument("--n_Gaussian", type=int, default=64)
parser.add_argument("--node_activation", type=str, default="Sigmoid")
parser.add_argument("--MLP_activation", type=str, default="Elu")
parser.add_argument("--use_node_batch_norm", type=bool, default=True)
parser.add_argument("--use_edge_batch_norm", type=bool, default=True)
parser.add_argument("--cutoff", type=int, default=5)
parser.add_argument("--N_block", type=int, default=6)
parser.add_argument("--n_hidden_feat", type=int, default=128, help='the dimension of node features')
parser.add_argument("--conv_bias", type=bool, default=False, help='use bias item or not in the linear layer')
args = parser.parse_args()


def set_seed(seed):
    """Set complete random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cal_train(loader, model, optimizer, device):
    """Fixed training function"""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for input, targets, ids in loader:
        optimizer.zero_grad()

        # Batch device transfer
        cs_x = input[0].to(device)
        in_x = input[1].to(device)
        cs_edge_index = input[2].to(device)
        in_edge_source = input[3].to(device)
        in_edge_target = input[4].to(device)
        cs_edge_attr = input[5].to(device)
        in_edge_attr = input[6].to(device)
        global_attr = input[7].to(device)
        cs_node_batch = input[8].to(device)
        in_node_batch = input[9].to(device)
        targets = targets.to(device)

        out = model(cs_x, in_x, cs_edge_index, in_edge_source, in_edge_target, cs_edge_attr, in_edge_attr,
                    global_attr, cs_node_batch, in_node_batch)

        # Calculate loss only once
        loss = F.l1_loss(out, targets)
        loss.backward()
        optimizer.step()

        # Accumulate loss (calculate average loss using mean reduction)
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    # Return average loss for entire epoch (i.e., train_mae)
    return total_loss / total_samples


def cal_eval(loader, model, device):
    """Fixed evaluation function"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for input, targets, ids in loader:
            # Device transfer
            cs_x = input[0].to(device)
            in_x = input[1].to(device)
            cs_edge_index = input[2].to(device)
            in_edge_source = input[3].to(device)
            in_edge_target = input[4].to(device)
            cs_edge_attr = input[5].to(device)
            in_edge_attr = input[6].to(device)
            global_attr = input[7].to(device)
            cs_node_batch = input[8].to(device)
            in_node_batch = input[9].to(device)
            targets = targets.to(device)

            out = model(cs_x, in_x, cs_edge_index, in_edge_source, in_edge_target, cs_edge_attr, in_edge_attr,
                        global_attr, cs_node_batch, in_node_batch)

            # Calculate loss
            loss = F.l1_loss(out, targets, reduction='sum')
            total_loss += loss.item()
            total_samples += targets.size(0)

            # Collect prediction results
            all_outputs.extend(out.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate R²
    all_outputs = np.array(all_outputs).flatten()
    all_targets = np.array(all_targets).flatten()
    r2 = r2_score(all_targets, all_outputs)

    # Return average loss and R²
    return total_loss / total_samples, r2


def save_data(loader, model, device):
    """Fixed data saving function"""
    model.eval()
    all_outputs = []
    all_targets = []
    all_ids = []

    with torch.no_grad():
        for input, targets, ids in loader:
            # Device transfer
            cs_x = input[0].to(device)
            in_x = input[1].to(device)
            cs_edge_index = input[2].to(device)
            in_edge_source = input[3].to(device)
            in_edge_target = input[4].to(device)
            cs_edge_attr = input[5].to(device)
            in_edge_attr = input[6].to(device)
            global_attr = input[7].to(device)
            cs_node_batch = input[8].to(device)
            in_node_batch = input[9].to(device)
            targets = targets.to(device)

            out = model(cs_x, in_x, cs_edge_index, in_edge_source, in_edge_target, cs_edge_attr, in_edge_attr,
                        global_attr, cs_node_batch, in_node_batch)

            # Collect data
            outputs = out.cpu().numpy().flatten()
            targets_np = targets.cpu().numpy().flatten()

            all_outputs.extend(outputs)
            all_targets.extend(targets_np)
            all_ids.extend(ids)

    return np.array(all_outputs), np.array(all_targets), all_ids


def cs_in_collate(batch):
    """Original collate function remains unchanged"""
    batch_cs_x = []
    batch_in_x = []

    batch_cs_edge_source = []
    batch_cs_edge_target = []
    batch_cs_edge_index = []
    batch_global_attr = []

    batch_in_edge_source = []
    batch_in_edge_target = []

    batch_cs_edge_attr = []
    batch_in_edge_attr = []

    batch_targets = []
    batch_ids = []

    cs_node_batch = []
    in_node_bacth = []

    cs_total_count, in_total_count = 0, 0
    for i, (cs_x, in_x, cs_edge_index, in_edge_index, cs_edge_attr, in_edge_attr, global_attr, targets,
            ids) in enumerate(batch):
        # Node feature
        batch_cs_x.append(cs_x)
        batch_in_x.append(in_x)
        # Edge index
        cs_edge_source = cs_edge_index[0]
        cs_edge_target = cs_edge_index[1]
        in_edge_source = in_edge_index[0]
        in_edge_target = in_edge_index[1]
        batch_cs_edge_source.append(cs_edge_source + cs_total_count)
        batch_cs_edge_target.append(cs_edge_target + cs_total_count)
        batch_in_edge_source.append(in_edge_source + in_total_count)
        batch_in_edge_target.append(in_edge_target + in_total_count)
        # Edge feature
        batch_cs_edge_attr.append(cs_edge_attr)
        batch_in_edge_attr.append(in_edge_attr)
        # global attribution
        batch_global_attr.append(global_attr)

        batch_targets.append(targets)
        batch_ids.append(ids)

        cs_node_batch += [i] * len(cs_x)
        in_node_bacth += [i] * len(in_x)
        cs_total_count += len(cs_x)
        in_total_count += len(in_x)

    batch_cs_x = np.concatenate(batch_cs_x, axis=0)
    batch_in_x = np.concatenate(batch_in_x, axis=0)
    batch_cs_edge_attr = np.concatenate(batch_cs_edge_attr, axis=0)
    batch_in_edge_attr = np.concatenate(batch_in_edge_attr, axis=0)
    batch_cs_edge_source = np.concatenate(batch_cs_edge_source, axis=0)
    batch_cs_edge_target = np.concatenate(batch_cs_edge_target, axis=0)
    batch_in_edge_source = np.concatenate(batch_in_edge_source, axis=0)
    batch_in_edge_target = np.concatenate(batch_in_edge_target, axis=0)
    batch_global_attr = np.concatenate(batch_global_attr, axis=0).reshape(-1, 3)
    batch_targets = np.concatenate(batch_targets, axis=0)
    batch_cs_edge_index.append(batch_cs_edge_source)
    batch_cs_edge_index.append(batch_cs_edge_target)

    batch_cs_x = torch.Tensor(batch_cs_x)
    batch_in_x = torch.Tensor(batch_in_x)
    batch_cs_edge_index = torch.LongTensor(batch_cs_edge_index)
    batch_in_edge_source = torch.LongTensor(batch_in_edge_source)
    batch_in_edge_target = torch.LongTensor(batch_in_edge_target)
    batch_cs_edge_attr = torch.Tensor(batch_cs_edge_attr)
    batch_in_edge_attr = torch.Tensor(batch_in_edge_attr)
    batch_global_attr = torch.Tensor(batch_global_attr)
    batch_targets = torch.Tensor(batch_targets)
    cs_node_batch = torch.LongTensor(cs_node_batch)
    in_node_bacth = torch.LongTensor(in_node_bacth)

    return (batch_cs_x, batch_in_x, \
            batch_cs_edge_index, \
            batch_in_edge_source, batch_in_edge_target, \
            batch_cs_edge_attr, batch_in_edge_attr, batch_global_attr, \
            cs_node_batch, in_node_bacth), batch_targets, batch_ids


def save_epoch_results(filename, fold, seed, epoch, train_mae, val_mae, r2_val, test_mae, r2_test, lr, batch_size):
    """
    Modified save function: Save each epoch's results to specified file, appending one row each time.

    Args:
        filename (str): Results file path.
        fold (int): Current cross-validation fold number.
        ... (other metrics)
    """
    import pandas as pd
    import os
    import time

    # Build result dictionary
    result_dict = {
        'fold': [fold],
        'seed': [seed],
        'epoch': [epoch],
        'train_mae': [train_mae],  # Average loss after each epoch training completes
        'val_mae': [val_mae],
        'r2_val': [r2_val],
        'test_mae': [test_mae],
        'r2_test': [r2_test],
        'lr': [lr],
        'batch_size': [batch_size],
        'timestamp': [time.strftime("%Y%m%d_%H%M%S")]
    }

    df = pd.DataFrame(result_dict)

    # If file doesn't exist, create and write header; otherwise append
    if not os.path.exists(filename):
        df.to_csv(filename, index=False, mode='w')
    else:
        df.to_csv(filename, index=False, mode='a', header=False)


if __name__ == '__main__':

    args.dataset_path = args.dataset
    print("Starting to process dataset:", args.dataset_path)

    dataset = load_data.load_data(args.dataset_path)

    # Store all fold results (only for calculating cross-validation standard deviation)
    all_fold_results = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Set fixed filename for epoch results to record train/val/test MAE for each epoch
    epoch_results_filename = f"mid_out/all_epoch_results_{timestamp}_bs{args.batch_size}_SDCGNN.csv"

    # Set feature dimensions (only needs to be set once)
    args.cs_x_features_dim = dataset[0][0].shape[1]
    args.cs_edge_features_dim = dataset[0][4].shape[1]
    args.in_x_features_dim = dataset[0][1].shape[1]
    args.in_edge_features_dim = dataset[0][5].shape[1]
    args.global_features_dim = dataset[0][6].shape[0]

    print(f"Dataset features - CS node: {args.cs_x_features_dim}, "
          f"CS edge: {args.cs_edge_features_dim}, "
          f"IN node: {args.in_x_features_dim}, "
          f"IN edge: {args.in_edge_features_dim}, "
          f"Global: {args.global_features_dim}")

    for fold in range(1, 11):
        # Set random seed
        seed = fold ** 2
        set_seed(seed)
        args.seed = seed

        # Shuffle and split dataset
        random.shuffle(dataset)
        train_set, test_val = train_test_split(dataset, train_size=0.8, random_state=seed)
        val_set, test_set = train_test_split(test_val, test_size=0.5, random_state=seed)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=cs_in_collate)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=cs_in_collate)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=cs_in_collate)

        print(f"\n=== Fold {fold} (Seed: {seed}) ===")
        print(f"train-val-test: {len(train_set)}-{len(val_set)}-{len(test_set)}")

        # Initialize model and optimizer
        model = SDCGNN(args).to(args.device)
        # model = MEGNET_Conv(args).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)

        t = time.time()
        best_val_loss = float('inf')

        # Training loop
        for epoch in range(args.epochs):
            train_loss = cal_train(train_loader, model, optimizer, args.device)
            val_loss, r2_val = cal_eval(val_loader, model, args.device)
            test_loss, r2_test = cal_eval(test_loader, model, args.device)
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()

            # Print logs
            print('Epoch:{:03d}'.format(epoch),
                  'train_mae:{:.3f}'.format(train_loss),
                  'val_mae:{:.3f}'.format(val_loss),
                  'r2_val:{:.3f}'.format(r2_val),
                  'test_mae:{:.3f}'.format(test_loss),
                  'r2_test:{:.3f}'.format(r2_test),
                  'lr:{:.6f}'.format(current_lr),
                  'time:{:.3f}'.format(time.time() - t))

            # Modified save logic: Save results after each epoch ends
            save_epoch_results(
                filename=epoch_results_filename,  # Use fixed filename
                fold=fold,
                seed=seed,
                epoch=epoch + 1,  # Let epoch count start from 1 for better readability
                train_mae=train_loss,
                val_mae=val_loss,
                r2_val=r2_val,
                test_mae=test_loss,
                r2_test=r2_test,
                lr=current_lr,
                batch_size=args.batch_size
            )

            # Update best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"mid_out/best_model_fold{fold}_seed{seed}.pth")  # Optimize model save path

        # Save final prediction results (only keep id, targets, prediction, remove prediction/true value standard deviation)
        all_train_outputs, all_train_targets, all_train_ids = save_data(train_loader, model, args.device)
        df_predictions = pd.DataFrame(
            {"id": all_train_ids, "targets": all_train_targets, "prediction": all_train_outputs})
        df_predictions.to_csv(f"mid_out/train_predictions_fold{fold}_{timestamp}_bs{args.batch_size}_SDCGNN.csv",
                              index=False)

        all_val_outputs, all_val_targets, all_val_ids = save_data(val_loader, model, args.device)
        df_predictions = pd.DataFrame({"id": all_val_ids, "targets": all_val_targets, "prediction": all_val_outputs})
        df_predictions.to_csv(f"mid_out/val_predictions_fold{fold}_{timestamp}_bs{args.batch_size}_SDCGNN.csv",
                              index=False)

        all_test_outputs, all_test_targets, all_test_ids = save_data(test_loader, model, args.device)
        df_predictions = pd.DataFrame({"id": all_test_ids, "targets": all_test_targets, "prediction": all_test_outputs})
        df_predictions.to_csv(f"mid_out/test_predictions_fold{fold}_{timestamp}_bs{args.batch_size}_SDCGNN.csv",
                              index=False)

        # Save each fold's final test results (only keep core fields, cross-validation standard deviation calculated later)
        final_test_loss, final_r2_test = cal_eval(test_loader, model, args.device)
        all_fold_results.append({
            'fold': fold,
            'seed': seed,
            'test_mae': final_test_loss,
            'test_r2': final_r2_test,
            'best_val_mae': best_val_loss
        })

        print(f"Fold {fold} completed - Test MAE: {final_test_loss:.4f}, R²: {final_r2_test:.4f}")

    # Summarize all fold results
    results_df = pd.DataFrame(all_fold_results)
    results_df.to_csv(f"mid_out/cross_validation_results_{timestamp}_SDCGNN.csv", index=False)

    # Calculate cross-validation average and standard deviation (core: dispersion between different folds)
    avg_test_mae = np.mean([r['test_mae'] for r in all_fold_results])
    avg_test_r2 = np.mean([r['test_r2'] for r in all_fold_results])
    avg_best_val_mae = np.mean([r['best_val_mae'] for r in all_fold_results])

    # Cross-validation standard deviation
    std_cv_test_mae = np.std([r['test_mae'] for r in all_fold_results], ddof=1)  # ddof=1 means sample standard deviation
    std_cv_test_r2 = np.std([r['test_r2'] for r in all_fold_results], ddof=1)
    std_cv_val_mae = np.std([r['best_val_mae'] for r in all_fold_results], ddof=1)

    print(f"\n=== Cross Validation Summary ===")
    print(f"Average Test MAE: {avg_test_mae:.4f} ± {std_cv_test_mae:.4f}")
    print(f"Average Test R²: {avg_test_r2:.4f} ± {std_cv_test_r2:.4f}")
    print(f"Average Best Val MAE: {avg_best_val_mae:.4f} ± {std_cv_val_mae:.4f}")

    # Save summary statistics (including averages and cross-validation standard deviations)
    summary_stats = pd.DataFrame({
        'metric': ['avg_test_mae', 'avg_test_r2', 'avg_best_val_mae',
                   'cv_test_mae_std', 'cv_test_r2_std', 'cv_val_mae_std'],
        'value': [avg_test_mae, avg_test_r2, avg_best_val_mae,
                  std_cv_test_mae, std_cv_test_r2, std_cv_val_mae]
    })
    summary_stats.to_csv(f"mid_out/cv_summary_stats_{timestamp}_SDCGNN.csv", index=False)

    print("\nAll data (including cross-validation standard deviations) have been saved successfully!")
