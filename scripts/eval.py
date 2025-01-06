#!/usr/bin/env python

import os
import argparse
import pickle
import torch
from os.path import isfile

from transloc4d.misc import TrainingParams 
from transloc4d.datasets import WholeDataset
from transloc4d.models import get_model
from transloc4d import evaluate_4drad_dataset, save_recall_results


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on a dataset.")
    parser.add_argument("--database_pickle", required=True, help="Path to the database pickle file")
    parser.add_argument("--query_pickle", required=True, help="Path to the query pickle file")
    parser.add_argument("--config", default = "../config/train/ntu-rsvi.txt", help="Path to the configuration file")
    parser.add_argument("--model_config", default = None, help="Path to the model-specific configuration file")
    parser.add_argument("--weights", required=True, help="Path to the trained model weights")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    
    args = parser.parse_args()


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    if args.model_config is None:
        model_folder = os.path.dirname(args.weights)
        model_config_path = os.path.join(model_folder, "model_config.txt")
        assert isfile(model_config_path), f"Model configuration file not found at {model_folder}, please provide the path to the model configuration file"
        args.model_config = model_config_path
    
    print(f"==> Loaded model parameters from {args.model_config}")

    params = TrainingParams(args.config, args.model_config, debug=False)  # Adjust parameters as necessary
    model = get_model(params, device, args.weights)

    test_set = WholeDataset(params.dataset_folder, args.database_pickle, args.query_pickle)

    # Run the evaluation
    recall_metrics = evaluate_4drad_dataset(model, device, test_set, params)

    for scene, scene_recall in recall_metrics.items():
        print(f"Scene: {scene}:")
        for n in [1, 5, 10]:
            recall_at_n = scene_recall['recall_at_n'].get(n, 0.0)  # 获取 recall@N，默认值为 0.0
            print(f"Recall@{n}: {recall_at_n:.4f}")

        print(f"Top 1% Recall: {scene_recall['recall_at_1_percent']:.4f}")
        print("\n")

    model_name = os.path.basename(args.weights).split('.')[0]
    result_dir = os.path.dirname(args.weights)
    database_name = os.path.basename(args.database_pickle).split('.')[0]
    query_name = os.path.basename(args.query_pickle).split('.')[0]

    dataset_name = f"{database_name}_{query_name}"
    save_recall_results(model_name, dataset_name, recall_metrics, result_dir)