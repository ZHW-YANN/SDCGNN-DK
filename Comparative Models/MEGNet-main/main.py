import argparse
import csv
import json
import os
import time
import pandas as pd
import torch
import torch.nn.functional as F
from data_loader import load_data
from torch.utils.data import DataLoader
from model import CS_Conv, IN_Conv, Merge_Model, CGCNN_Conv, IN_CGCNN_Conv, GATGNN_Conv, GeoCGNN_Conv, MEGNET_Conv
import numpy as np
import random
import csv
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=666, help='random_seed')
parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
parser.add_argument('--nhid', type=int, default=128, help='dimension')
parser.add_argument('--lr', type=float, default=0.0001, help='learning_rate')
parser.add_argument('--weight_decay1', type=float, default=0, help='weight_decay')
parser.add_argument('--weight_decay2', type=float, default=0, help='weight_decay')
parser.add_argument('--weight_decay3', type=float, default=0, help='weight_decay')
parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
parser.add_argument('--device', type=str, default='cuda', help='training devices')
parser.add_argument('--dataset', type=str, default='data', help='the path of dataset')
parser.add_argument('--neighbors', type=int, default=12, help='the max number of neighbors')
parser.add_argument('--num_classes', type=int, default=3, help='the number of classes')
parser.add_argument('--cat_or_add', type=str, default="cat")
parser.add_argument('--a', type=float, default=0.5)
parser.add_argument('--b', type=float, default=0.5)
parser.add_argument("--n_MLP_LR", type=int, default=2)
parser.add_argument("--n_grid_K", type=int, default=4)
parser.add_argument("--n_Gaussian", type=int, default=64)
parser.add_argument("--node_activation", type=str, default="Sigmoid")
parser.add_argument("--MLP_activation", type=str, default="Elu")
parser.add_argument("--use_node_batch_norm", type=bool, default=True)
parser.add_argument("--use_edge_batch_norm", type=bool, default=True)
parser.add_argument("--cutoff", type=int, default=8)
parser.add_argument("--N_block", type=int, default=6)
parser.add_argument("--n_hidden_feat", type=int, default=128, help='the dimension of node features')
parser.add_argument("--conv_bias", type=bool, default=False, help='use bias item or not in the linear layer')
args = parser.parse_args()

def cal_CGCNN_train(loader):
    cs_model.train()
    loss_train = 0.0
    for input, targets, ids in loader:
        cs_optimizer.zero_grad()
        cs_x = input[0].to(args.device)
        cs_edge_index = input[1].to(args.device)
        # cs_edge_targets = input[2].to(args.device)
        cs_edge_attr = input[2].to(args.device)
        cs_global_attr = input[3].to(args.device)
        cs_node_batch = input[4].to(args.device)
        targets = targets.to(args.device)
        out = cs_model(cs_x, cs_edge_index, cs_edge_attr, cs_global_attr, cs_node_batch)
        loss = F.l1_loss(out, targets)
        loss.backward()
        cs_optimizer.step()
        loss_train += F.l1_loss(out, targets, reduction='sum').item()
    return loss_train


def cal_CGCNN_eval(loader):
    cs_model.eval()
    loss_eval = 0.0
    testResult = []
    true_value = []
    with torch.no_grad():
        for input, targets, ids in loader:
            cs_optimizer.zero_grad()
            cs_x = input[0].to(args.device)
            cs_edge_index = input[1].to(args.device)
            cs_edge_attr = input[2].to(args.device)
            cs_global_attr = input[3].to(args.device)
            cs_node_batch = input[4].to(args.device)
            targets = targets.to(args.device)
            out = cs_model(cs_x, cs_edge_index, cs_edge_attr, cs_global_attr, cs_node_batch)
            loss_eval += F.l1_loss(out, targets, reduction='sum').item()
            for i in range(len(out)):
                testResult.append(out[i].cpu().numpy())
                true_value.append(targets[i].cpu().numpy())
        r2 = round(r2_score(true_value, testResult), 4)
    return loss_eval, r2


def save_data(loader):
    cs_model.eval()
    loss_eval = 0.0
    all_outputs = []
    all_targets = []
    all_ids = []
    with torch.no_grad():
        for input, targets, ids in loader:
            cs_optimizer.zero_grad()
            cs_x = input[0].to(args.device)
            cs_edge_index = input[1].to(args.device)
            cs_edge_attr = input[2].to(args.device)
            cs_global_attr = input[3].to(args.device)
            cs_node_batch = input[4].to(args.device)

            targets = targets.to(args.device)
            out = cs_model(cs_x, cs_edge_index, cs_edge_attr, cs_global_attr, cs_node_batch)
            loss_eval += F.l1_loss(out, targets, reduction='sum').item()
            outputs = out.to(torch.device("cpu")).numpy()
            targets = targets.to(torch.device("cpu")).numpy()
            all_outputs.append(outputs)
            all_targets.append(targets)
            all_ids.extend(ids)
        all_outputs = np.concatenate(all_outputs)
        all_targets = np.concatenate(all_targets)
    return all_outputs, all_targets, all_ids


def meg_collate(batch):
    batch_cs_x = []
    batch_cs_edge_source = []
    batch_cs_edge_target = []
    batch_cs_edge_index = []
    batch_cs_edge_attr = []
    batch_cs_global_attr = []
    batch_targets = []
    batch_ids = []
    cs_node_batch = []
    cs_total_count = 0
    for i, (cs_x, cs_edge_index, cs_edge_attr, cs_global_attr, targets, ids) in enumerate(batch):
        batch_cs_x.append(cs_x)

        cs_edge_source = cs_edge_index[0]
        cs_edge_target = cs_edge_index[1]

        batch_cs_edge_source.append(cs_edge_source + cs_total_count)
        batch_cs_edge_target.append(cs_edge_target + cs_total_count)

        batch_cs_edge_attr.append(cs_edge_attr)
        batch_cs_global_attr.append(cs_global_attr)

        batch_targets.append(targets)
        batch_ids.append(ids)

        cs_node_batch += [i] * len(cs_x)
        cs_total_count += len(cs_x)

    batch_cs_x = np.concatenate(batch_cs_x, axis=0)
    batch_cs_edge_attr = np.concatenate(batch_cs_edge_attr, axis=0)
    batch_cs_edge_source = np.concatenate(batch_cs_edge_source, axis=0)
    batch_cs_edge_target = np.concatenate(batch_cs_edge_target, axis=0)
    batch_cs_global_attr = np.concatenate(batch_cs_global_attr, axis=0).reshape(-1, 2)
    batch_targets = np.concatenate(batch_targets, axis=0)
    batch_cs_edge_index.append(batch_cs_edge_source)
    batch_cs_edge_index.append(batch_cs_edge_target)

    batch_cs_x = torch.Tensor(batch_cs_x)
    # batch_cs_edge_source = torch.LongTensor(batch_cs_edge_source)
    # batch_cs_edge_target = torch.LongTensor(batch_cs_edge_target)
    batch_cs_edge_index = torch.LongTensor(batch_cs_edge_index)
    batch_cs_edge_attr = torch.Tensor(batch_cs_edge_attr)
    batch_cs_global_attr = torch.Tensor(batch_cs_global_attr)
    batch_targets = torch.Tensor(batch_targets)
    cs_node_batch = torch.LongTensor(cs_node_batch)

    return (batch_cs_x, batch_cs_edge_index, batch_cs_edge_attr, batch_cs_global_attr, cs_node_batch), batch_targets, batch_ids


if __name__ == '__main__':
    args.dataset_path = args.dataset
    dataset = load_data.load_data_MEGNET(args.dataset_path)
    for i in range(1, 11):
        args.seed = i

        train_set, val_set = train_test_split(dataset, train_size=0.8, random_state=args.seed)
        val_set, test_set = train_test_split(val_set, train_size=0.5, random_state=args.seed)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=meg_collate)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=meg_collate)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=meg_collate)

        print("train-val-test:", len(train_set), '-', len(val_set), '-', len(test_set))
        args.cs_x_features_dim = dataset[0][0].shape[1]
        args.cs_edge_features_dim = dataset[0][2].shape[1]
        args.cs_global_features_dim = dataset[0][3].shape[0]

        print(f"===============================================seed{args.seed}===============================================")
        # 1、晶体结构模型训练
        print("Crystal structure model starts training!")
        cs_model = MEGNET_Conv.MEGNet(args).to(args.device)
        cs_optimizer = torch.optim.Adam(cs_model.parameters(), lr=args.lr, weight_decay=args.weight_decay1)
        cs_scheduler = torch.optim.lr_scheduler.StepLR(cs_optimizer, step_size=80, gamma=0.1)
        t = time.time()
        for epoch in range(args.epochs):
            train_loss = cal_CGCNN_train(train_loader)
            val_loss, r2_val = cal_CGCNN_eval(val_loader)
            test_loss, r2_test = cal_CGCNN_eval(test_loader)
            cs_scheduler.step()
            print('Epoch:{:03d}'.format(epoch),
                  'train_mae:{:3f}'.format(train_loss / len(train_set)),
                  'val_mae:{:3f}'.format(val_loss / len(val_set)),
                  'r2_val:{:3f}'.format(r2_val),
                  'test_mae:{:3f}'.format(test_loss / len(test_set)),
                  'r2_test:{:3f}'.format(r2_test),
                  'lr:{:.6f}'.format(cs_optimizer.param_groups[0]['lr']),
                  'time:{:.3f}'.format(time.time() - t))
        all_train_outputs, all_train_targets, all_train_ids = save_data(train_loader)
        df_predictions = pd.DataFrame({"id": all_train_ids, "targets": all_train_targets, "prediction": all_train_outputs})
        df_predictions.to_csv(f"mid_out/train_predictions_{args.seed}.csv", index=False)

        all_val_outputs, all_val_targets, all_val_ids = save_data(val_loader)
        df_predictions = pd.DataFrame({"id": all_val_ids, "targets": all_val_targets, "prediction": all_val_outputs})
        df_predictions.to_csv(f"mid_out/val_predictions_{args.seed}.csv", index=False)

        all_test_outputs, all_test_targets, all_test_ids = save_data(test_loader)
        df_predictions = pd.DataFrame({"id": all_test_ids, "targets": all_test_targets, "prediction": all_test_outputs})
        df_predictions.to_csv(f"mid_out/test_predictions_{args.seed}.csv", index=False)

        print("all data have saved successfully!")


