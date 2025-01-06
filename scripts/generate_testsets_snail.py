import os
import pickle
import numpy as np
import pandas as pd
import argparse
import tqdm
from sklearn.neighbors import KDTree


##########################################
# split query and database data
# save in evaluation_database.pickle / evaluation_query.pickle
##########################################

def output_to_file(output, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved: {filename}")


def construct_query_and_database_sets(base_path, data_type, seqs, positive_dist, yaw_threshold, use_yaw,
                                      use_timestamp_name):
    # Initialize empty dictionaries for database and query sets
    database_sets = {}
    query_sets = {}

    # Process each sequence
    for seq_id, seq in enumerate(tqdm.tqdm(seqs)):
        seq_path = os.path.join(base_path, data_type, seq)
        tras = [name for name in os.listdir(seq_path) if os.path.isdir(os.path.join(seq_path, name))]
        tras.sort()

        # Initialize temporary database and query lists for each sequence
        database = []
        query = []

        # Process each trajectory within a sequence
        for tra_id, tra in enumerate(tras):
            pose_path = os.path.join(seq_path, f"{tra}_poses.txt")
            df_locations = pd.read_table(
                pose_path, sep=' ', converters={'timestamp': str},
                names=['timestamp', 'r11', 'r12', 'r13', 'x', 'r21', 'r22', 'r23', 'y', 'r31', 'r32', 'r33', 'z', 'yaw']
            )
            df_locations = df_locations[['timestamp', 'x', 'y', 'z', 'yaw']]

            # Create file names for data points
            if use_timestamp_name:
                df_locations['file'] = df_locations['timestamp'].apply(
                    lambda t: os.path.join(seq_path, tra, f"{t}.bin")
                )
            else:
                df_locations['file'] = range(0, len(df_locations))
                df_locations['file'] = df_locations['file'].apply(lambda x: str(x).zfill(6))
                df_locations['file'] = df_locations['file'].apply(lambda x: os.path.join(data_type, seq, tra, str(x) + '.bin'))

            # Database (first trajectory of each sequence)
            if tra_id == 0:
                df_database = df_locations
                for _, row in df_database.iterrows():
                    database.append({
                        "file": row['file'],
                        "northing": row['x'],
                        "easting": row['y'],
                        "yaw": row['yaw'],
                        "scene": seq
                    })
                # Build KDTree for efficient querying
                database_tree = KDTree(df_database[['x', 'y']])

            else:  # Query (subsequent trajectories)
                df_query = df_locations
                for _, row in df_query.iterrows():
                    coor = np.array([[row['x'], row['y']]])
                    indices = database_tree.query_radius(coor, r=positive_dist)[0].tolist()

                    # Adjust indices by adding yaw condition if needed
                    if use_yaw:
                        yaw_diff = np.abs(row['yaw'] - df_database.iloc[indices]['yaw'].values)
                        indices = [idx for idx, diff in zip(indices, yaw_diff) if
                                   np.min((diff, 360 - diff)) < yaw_threshold]

                    query.append({
                        "file": row['file'],
                        "northing": row['x'],
                        "easting": row['y'],
                        "yaw": row['yaw'],
                        "positives": indices,
                        "scene": seq
                    })

        # Store the database and query data for this scene (seq)
        database_sets[seq] = database
        query_sets[seq] = query

    # Save output as pickle files
    save_folder = os.path.join(base_path, 'radar_split/')
    os.makedirs(save_folder, exist_ok=True)

    # Save database to file
    database_file = os.path.join(save_folder, f"evaluation_database_{data_type}.pickle")
    output_to_file(database_sets, database_file)

    # Save query to file
    query_file = os.path.join(save_folder,
                              f"evaluation_query_{data_type}_{positive_dist}m{'_' + str(yaw_threshold) if use_yaw else ''}.pickle")
    output_to_file(query_sets, query_file)


# Building database and query files for evaluation
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='/media/cyw/KESU/datasets/clean_radar/processed_snail_radar/oculii/',
                        help='radar datasets path')
    parser.add_argument('--data_type', type=str, default='test', help='test_short or test_long')
    parser.add_argument('--positive_dist', type=float, default=9,
                        help='Positive sample distance threshold, short:5, long:9')
    parser.add_argument('--yaw_threshold', type=float, default=30, help='Yaw angle threshold, 8 or 25')
    parser.add_argument('--use_yaw', type=bool, default=True, help='If use yaw to determine a positive sample.')
    parser.add_argument('--use_timestamp_name', type=bool, default=False, help='save most similar index for loss')
    cfgs = parser.parse_args()

    # Define sequences based on the data type and dataset
    if 'valid' in cfgs.data_type:
        if "ars548" in cfgs.data_path:
            seqs = ['if', 'iaf']  # , 'sl'
        elif "oculii" in cfgs.data_path:
            seqs = ['if', 'iaf']
        else:
            raise Exception('Loading error!')
    elif 'test' in cfgs.data_type:
        if "ars548" in cfgs.data_path:
            seqs = ['bc', 'ss', 'st', 'sl', 'if', 'iaf', 'iaef', '81r']
        elif "oculii" in cfgs.data_path:
            seqs = ['sl'] #, 'st', , 'if', 'iaf', 'iaef', '81r'
        else:
            raise Exception('Loading error!')

    # Call the function to construct query and database sets
    construct_query_and_database_sets(cfgs.data_path, cfgs.data_type, seqs, cfgs.positive_dist, cfgs.yaw_threshold,
                                      cfgs.use_yaw, cfgs.use_timestamp_name)
