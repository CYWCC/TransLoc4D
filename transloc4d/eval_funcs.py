import os
import torch
import numpy as np
import faiss
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import test_collate_fn
from collections import defaultdict

import faiss
import numpy as np
from collections import defaultdict


def get_predictions_by_scene(dataset, embeddings):

    query_groups = dataset.queries
    database_groups = dataset.database

    query_lengths = {scene: dataset.query_lengths[scene] for scene in query_groups.keys()}
    database_lengths = {scene: dataset.database_lengths[scene] for scene in database_groups.keys()}

    all_predictions = {}

    all_gt = [dataset.get_non_negatives(i) for i in range(len(dataset.queries_pc_file))]

    qFeat = embeddings[:dataset.len_q].cpu().numpy().astype("float32")
    dbFeat = embeddings[dataset.len_q:].cpu().numpy().astype("float32")

    query_start_idx = 0
    db_start_idx = 0

    for scene in query_groups.keys():
        query_scene_len = query_lengths[scene]
        database_scene_len = database_lengths[scene]

        threshold = max(int(round(database_scene_len / 100.0)), 1)
        num_neighbors = max(25, threshold)

        scene_gt = all_gt[query_start_idx:query_start_idx + query_lengths[scene]]

        query_embeddings = qFeat[query_start_idx:query_start_idx + query_scene_len]
        database_embeddings = dbFeat[db_start_idx:db_start_idx + database_scene_len]

        query_start_idx += query_scene_len
        db_start_idx += database_scene_len

        # 使用 Faiss 进行检索
        print(f"==> Building faiss index for scene: {scene}")
        faiss_index = faiss.IndexFlatL2(query_embeddings.shape[1])  # 使用 L2 距离
        faiss_index.add(database_embeddings)  # 将数据库 embeddings 添加到索引中

        # 为每个 query 进行检索，获得前 20 个最接近的数据库条目
        dis, predictions = faiss_index.search(query_embeddings, num_neighbors)

        # 保存当前场景的预测结果以及 ground truth
        all_predictions[scene] = {
            "predictions": predictions,
            "distances": dis,
            "gt": scene_gt  # 将对应的 ground truth 添加进来
        }

    return all_predictions


def get_predictions(dataset, embeddings):
    gt = []
    for i in range(dataset.len_q):
        positives = dataset.get_non_negatives(i)
        gt.append(positives)

    # get distance
    qFeat = embeddings[: dataset.len_q].cpu().numpy().astype("float32")
    dbFeat = embeddings[dataset.len_q :].cpu().numpy().astype("float32")

    print("==> Building faiss index")
    faiss_index = faiss.IndexFlatL2(qFeat.shape[1])
    faiss_index.add(dbFeat)
    dis, predictions = faiss_index.search(qFeat, 20)

    return predictions, gt


def evaluate_4drad_dataset(model, device, dataset, params):
    model.eval()
    quantizer = params.model_params.quantizer
    val_collate_fn = test_collate_fn(
        dataset,
        quantizer,
        params.batch_split_size,
        params.model_params.input_representation,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=params.val_batch_size,
        collate_fn=val_collate_fn,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=True,
    )
    embeddings_dataset = torch.empty((len(dataset), 256))
    with torch.no_grad():
        for iteration, batch in enumerate(tqdm(dataloader, desc="==> Computing embeddings")):
            embeddings_l = []
            for minibatch in batch:
                minibatch = {e: minibatch[e].to(device) for e in minibatch}
                ys = model(minibatch)
                embeddings_l.append(ys["global"])
                del ys

            embeddings = torch.cat(embeddings_l, dim=0)
            embeddings_dataset[
            iteration
            * params.val_batch_size: (iteration + 1)
                                     * params.val_batch_size
            ] = embeddings.detach()
            torch.cuda.empty_cache()

    predictions = get_predictions_by_scene(dataset, embeddings_dataset)
    recalls = compute_recall_by_scene(predictions)
    print("==> Evaluation completed!")

    return recalls

import numpy as np

def compute_recall_by_scene(predictions, n_values=[1, 5, 10, 20]):
    all_recalls = {}

    # 针对每个场景计算 recall
    for scene, result in predictions.items():
        scene_predictions = result["predictions"]  # 预测结果（每个查询的前n个最接近的数据库索引）
        scene_gt = result["gt"]  # ground truth
        num_queries = len(scene_predictions)

        # 用于计算每个场景的 recall_at_n
        correct_at_n = np.zeros(len(n_values))
        numQ = 0
        one_percent_retrieved = 0

        for qIx, pred in enumerate(scene_predictions):
            if len(scene_gt[qIx]) == 0:
                continue  # 如果没有正例，跳过
            else:
                numQ += 1

            # 计算 recall_at_n
            for i, n in enumerate(n_values):
                # 如果在 top-n 中找到了正例
                if np.any(np.in1d(pred[:n], scene_gt[qIx])):
                    correct_at_n[i:] += 1  # 如果命中，将所有后续的 n 也标记为正确
                    break  # 找到第一个命中的 n 值后跳出

            if len(list(set(scene_predictions[qIx]).intersection(set(scene_gt[qIx])))) > 0:
                one_percent_retrieved += 1

        recall_at_n = correct_at_n / numQ

        recall_at_1_percent = one_percent_retrieved / numQ if numQ > 0 else 0

        # 存储每个场景的 recall_at_n 和 recall@1%
        all_recalls[scene] = {
            "recall_at_n": {n: recall_at_n[i] for i, n in enumerate(n_values)},  # recall@1, recall@5, recall@10, recall@20
            "recall_at_1_percent": recall_at_1_percent  # recall@1%
        }

    return all_recalls


def save_recall_results(model_name, dataset_name, recall_metrics, result_dir, n_values=[1, 5, 10, 20]):
    """
    Save recall results to a file, including the recall for each scene and the average recall.

    Parameters:
    - model_name: The name of the model (used in the filename).
    - dataset_name: The name of the dataset.
    - recall_metrics: The recall metrics, which includes the per-scene recalls and the average recall.
    - result_dir: The directory where the result file will be saved.
    - n_values: List of recall@N values to be saved (default [1, 5, 10, 20]).
    """
    # Create the directory for the results if it doesn't exist
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Construct the filename and filepath
    filename = f"{model_name}_recall_results.txt"
    filepath = os.path.join(result_dir, filename)

    # Initialize the string to hold all recall results
    recall_results_str = f"Model: {model_name}, Dataset: {dataset_name}\n"

    # Iterate through each scene's recall metrics and add them to the string
    for scene, scene_recall in recall_metrics.items():
        recall_results_str += f"\nScene: {scene}\n"
        recall_results_str += "Recall @N:\n"
        for n in n_values:
            recall_results_str += f"Recall @{n}: {scene_recall['recall_at_n'].get(n, 0.0):.4f}\n"
        recall_results_str += f"Top 1% Recall: {scene_recall['recall_at_1_percent']:.4f}\n"

    # Calculate the average recall across all scenes
    avg_recall = {n: np.mean([recall['recall_at_n'].get(n, 0.0) for recall in recall_metrics.values()]) for n in
                  n_values}
    avg_top_1_percent_recall = np.mean([recall['recall_at_1_percent'] for recall in recall_metrics.values()])

    # Append the average recall information
    recall_results_str += "\nAverage Recall @N:\n"
    for n in n_values:
        recall_results_str += f"Recall @{n}: {avg_recall[n]:.4f}\n"
    recall_results_str += f"Average Top 1% Recall: {avg_top_1_percent_recall:.4f}\n"

    # Write the recall results to the file
    with open(filepath, 'a') as file:
        file.write(recall_results_str)

    print(f"==> Results saved to {filepath}")