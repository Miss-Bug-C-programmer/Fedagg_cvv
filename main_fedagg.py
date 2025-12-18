# main_fedagg.py
import os
import argparse
import logging
import numpy as np
import torch
import csv
from model_zoo import create_model
from fedagg import run_fedagg
from data_loader import load_partition_data_cifar10
import random

def add_args(parser):
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--comm_round', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--client_number', type=int, default=4)
    parser.add_argument('--edge_number', type=int, default=2)
    parser.add_argument('--partition_method', type=str, default='hetero')
    parser.add_argument('--partition_alpha', type=float, default=3.0)
    parser.add_argument('--method', type=str, default='fedagg')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--class_num', type=int, default=10)
    parser.add_argument("--exp_id", type=int, default=1)
    parser.add_argument("--csv_dir", type=str, default="experiment_results")
    parser.add_argument("--overwrite_csv", action="store_true")

    # 新增CVV参数
    parser.add_argument("--sync_strategy", type=str, default="full")
    parser.add_argument("--T_sync", type=int, default=1)
    parser.add_argument("--delta_v", type=int, default=0)

    return parser.parse_args()

def load_data(args):
    return load_partition_data_cifar10(args.dataset, args.data_dir,
                                       args.partition_method,
                                       args.partition_alpha,
                                       args.client_number, args.batch_size)

def create_edge_models(args, n_classes):
    random.seed(456)
    return [create_model("resnet10") for _ in range(args.edge_number)]

def create_client_models(args, n_classes):
    random.seed(123)
    return [create_model("cnn") for _ in range(args.client_number)]

def create_cloud_model(args, n_classes):
    return create_model("resnet18")


if __name__ == "__main__":
    args = add_args(argparse.ArgumentParser())
    logging.info(args)

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(5))

    dataset = load_data(args)
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, test_data_local_num_dict,
     train_data_local_dict, test_data_local_dict,
     class_num_train, class_num_test] = dataset

    client_models = create_client_models(args, class_num_train)
    edge_models = create_edge_models(args, class_num_train)
    cloud_model = create_cloud_model(args, class_num_train)

    metrics = run_fedagg(client_models, edge_models, cloud_model,
                         train_data_local_num_dict, test_data_local_num_dict,
                         train_data_local_dict, test_data_local_dict,
                         test_data_global, args)

    # 写 CSV
    os.makedirs(args.csv_dir, exist_ok=True)
    csv_path = os.path.join(args.csv_dir, f"cvv-fedagg_results_{args.dataset}.csv")
    if args.overwrite_csv and os.path.exists(csv_path):
        os.remove(csv_path)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exp_id", "round", "acc", "consistency",
                    "comm_cost_cloud", "comm_cost_edges", "comm_cost_total",
                    "chr_cloud", "chr_edges", "chr_ends"])
        for r in range(args.comm_round):
            acc = metrics["acc_list"][r]
            cons = metrics["consistency_list"][r]
            c_cloud = metrics["comm_cost_cloud_MB"][r]
            c_edges = metrics["comm_cost_edges_MB"][r]
            c_total = (c_cloud or 0) + (c_edges or 0)
            w.writerow([args.exp_id, r+1, acc, cons,
                        c_cloud, c_edges, c_total,
                        metrics["chr_cloud"][r], metrics["chr_edges"][r], metrics["chr_ends"][r]])
    print(f"[write] {csv_path} 已保存")
