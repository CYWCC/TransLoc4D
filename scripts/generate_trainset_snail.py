import os
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import KDTree
from transloc4d.datasets.base_datasets import TrainingTuple


def construct_query_dict(df_centroids, positive_dist, negative_dist):
    """
    This function is used to construct a query dictionary for training.
    It calculates the positive and negative samples for each query.
    """
    # Use KDTree to find nearest neighbors in the entire dataset, not limited to a single group
    tree = KDTree(df_centroids[['x', 'y']])
    ind_nn = tree.query_radius(df_centroids[['x', 'y']], r=positive_dist)  # Find positive samples
    ind_r = tree.query_radius(df_centroids[['x', 'y']], r=negative_dist)  # Find potential negative samples

    queries = {}
    for anchor_ndx in range(len(ind_nn)):
        data_id = int(df_centroids.iloc[anchor_ndx]["data_id"])
        data_file = df_centroids.iloc[anchor_ndx]["data_file"]
        yaw = df_centroids.iloc[anchor_ndx]["yaw"]
        timestamp = df_centroids.iloc[anchor_ndx]["timestamp"]

        # Find positive candidates (distance < positive_dist)
        positive_candis = np.setdiff1d(ind_nn[anchor_ndx], [anchor_ndx]).tolist()
        yaw_diff = np.abs(yaw - df_centroids.iloc[positive_candis]["yaw"])

        # Only keep positives that satisfy the yaw threshold
        positives = [c for c in positive_candis if np.min((yaw_diff[c], 360 - yaw_diff[c])) < cfgs.yaw_threshold]

        # Find all potential negative samples (distance < negative_dist)
        non_negative_candis = ind_r[anchor_ndx]
        non_negatives = np.sort(non_negative_candis)

        # Remove positives from the non_negatives list to ensure that they are truly negative
        negatives = np.setdiff1d(df_centroids.index.values.tolist(), non_negatives).tolist()

        queries[data_id] = TrainingTuple(
            id=data_id,
            timestamp=timestamp,
            rel_scan_filepath=data_file,
            positives=positives,
            non_negatives=non_negatives,
            position=df_centroids.iloc[anchor_ndx][['x', 'y']].values  # Assuming position is ['x', 'y', 'z']
        )

    return queries


def split_dataset(base_path, data_type, save_path, positive_dist, negative_dist, use_timestamp_name):
    """
    This function splits the dataset and creates training queries by finding positives and negatives.
    It now works across all groups, not just within a single group.
    """
    data_path = os.path.join(base_path, data_type)
    groups = sorted(os.listdir(data_path))
    train_seqs = {}

    all_centroids = []  # To hold all data for KDTree construction across all groups

    # First, collect all the data from all groups into one dataset
    for group_id, group in enumerate(tqdm(groups)):
        group_dir = os.path.join(data_path, group)
        seqs = [name for name in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, name))]

        for seq in tqdm(seqs):
            seq_poses_path = os.path.join(group_dir, seq + '_poses.txt')
            df_locations = pd.read_table(seq_poses_path, sep=' ', converters={'timestamp': str},
                                         names=['timestamp', 'r11', 'r12', 'r13', 'x', 'r21', 'r22', 'r23', 'y',
                                                'r31', 'r32', 'r33', 'z', 'yaw'])
            df_locations = df_locations.loc[:, ['timestamp', 'x', 'y', 'z', 'yaw']]

            if use_timestamp_name:
                df_locations['data_file'] = data_type + '/' + group + '/' + seq + '/' + df_locations[
                    'timestamp'] + '.bin'
            else:
                df_locations['data_id'] = range(0, len(df_locations))
                df_locations['data_file'] = df_locations['data_id'].apply(lambda x: str(x).zfill(6))
                df_locations['data_file'] = data_type + '/' + group + '/' + seq + '/' + df_locations[
                    'data_file'] + '.bin'

            all_centroids.append(df_locations)  # Append data from each sequence

    # Now we have all centroids, concatenate them to form a full dataset
    full_df = pd.concat(all_centroids, ignore_index=True)
    full_df['data_id'] = range(0, len(full_df))  # Update data_id for the full dataset

    # Construct query dicts across all groups
    queries = construct_query_dict(full_df, positive_dist, negative_dist)
    train_seqs.update(queries)

    # Save the split data to pickle
    with open(save_path, 'wb') as handle:
        pickle.dump(train_seqs, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/media/cyw/KESU/datasets/clean_radar/processed_snail_radar/oculii/', help='Dataset path')
    # parser.add_argument('--save_folder', type=str, default='./snail_split/', help='Path to save split data')
    parser.add_argument('--data_type', type=str, default='train', help='train or val')
    parser.add_argument('--positive_dist', type=float, default=9, help='Positive sample distance threshold')
    parser.add_argument('--negative_dist', type=float, default=18, help='Negative sample distance threshold')
    parser.add_argument('--yaw_threshold', type=float, default=75, help='Yaw angle threshold')
    parser.add_argument('--use_timestamp_name', type=bool, default=False, help='Save most similar index for loss')
    cfgs = parser.parse_args()

    save_name = f'training_queries_{cfgs.data_type}_{cfgs.positive_dist}m_{cfgs.yaw_threshold}.pickle'

    save_path = os.path.join(cfgs.data_path, save_name)
    split_dataset(cfgs.data_path, cfgs.data_type, save_path, cfgs.positive_dist,
                  cfgs.negative_dist, cfgs.use_timestamp_name)
    print(f"Completed: {save_path}")
