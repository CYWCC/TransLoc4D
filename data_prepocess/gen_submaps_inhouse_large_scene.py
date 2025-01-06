import numpy as np
import os
import rosbag
import argparse
from tqdm import tqdm
import sensor_msgs.point_cloud2 as pc2
from tools import interpolation, load_poses, standardization, random_down_sample, standardization_pv
from scipy.spatial.transform import Rotation
import open3d as o3d
import copy

# Constants for splitting the dataset
SPLIT_CONFIG = {
    # 'train': {'20231105':['2'],'20231105_aft': ['4','5'], '20231109': ['4'], '20231208': ['4'],
    #           '20240116': ['5'], '20240123': ['2']},
    # 'valid': { '20231201': ['2', '3'], '20231213': ['4'], '20240115': ['3']},
    'test': {
            # '20230920': ['1', '2'],  '20230921': ['2', '3', '4', '5'], '20231007': ['2', '4'],'20231019': ['1', '2'],  # only oculii
            '20231105': ['3', '4', '5', '6'],  '20231105_aft': ['2'],'20231208': ['1', '5'], '20231109': ['3'], '20231213': ['1', '2', '3', '5'],
            '20240113': ['1', '2', '3', '5'], '20240115': ['2'], '20240116': ['2', '4'], '20240116_eve': ['3', '4', '5'], '20240123': ['3']}
    }

# Dataset structure for different places and corresponding sequences
PLACE_SETUP_DICT = {
    'bc': ['20230920/1', '20230921/2', '20231007/4', '20231105/6','20231105_aft/2'],
    'sl': ['20230920/2', '20230921/3', '20230921/5', '20231007/2', '20231019/1', '20231105/2', '20231105/3', '20231109/3', '20231105_aft/4',],
    'ss': ['20230921/4', '20231019/2', '20231105/4', '20231105/5', '20231105_aft/5', '20231109/4'],
    'if': ['20231208/4', '20231213/4', '20231213/5', '20240115/3', '20240116/5', '20240116_eve/5', '20240123/3'],
    'iaf': ['20231201/2', '20231201/3', '20231208/5', '20231213/2', '20231213/3', '20240113/2', '20240113/3', '20240116_eve/4'],
    'iaef': ['20240113/5', '20240115/2', '20240116/4'],
    'st': ['20231208/1', '20231213/1', '20240113/1'],
    '81r': ['20240116/2', '20240116_eve/3', '20240123/2'],
}

def load_inhouse_calib(file_path):
    with open(file_path, 'r') as f:
        calib = np.loadtxt(f)
    return calib


def get_group_and_split(folder, seq_num, place_setup, split_config):
    """
    Return the group and split folder based on folder and sequence number.
    """
    for split, group_config in split_config.items():
        group_list = group_config.get(folder)
        if group_list and str(seq_num) in group_list:
            group_name = place_setup[group_list.index(str(seq_num))]
            return group_name, split
    return None, None


def process_bag_data(seq_path, gt_path, cfgs):
    """
    Process rosbag data and generate submap files.
    """
    bag = rosbag.Bag(seq_path, "r")
    bag_data = bag.read_messages(cfgs.radar_topic)

    pointclouds = []
    rcs_list = []
    doppler_list = []
    radar_times = []
    gt_times, gt_positions, gt_quats = load_poses(gt_path, ' ')
    min_gt_time, max_gt_time = min(gt_times), max(gt_times)

    for topic, msg, t in bag_data:
        time = float('%6f' % (msg.header.stamp.to_sec()))
        if time < min_gt_time or time > max_gt_time:
            continue

        # Extract point cloud data
        radar = pc2.read_points(msg, skip_nans=True,
                                field_names=('x', 'y', 'z', 's', 'intensity') if cfgs.use_rcs else (
                                'x', 'y', 'z', 's'))
        points = np.array(list(radar))

        if cfgs.use_doppler:
            doppler = copy.deepcopy(points[:, 3])  # Doppler velocity
            doppler_list.append(doppler)
            
        if cfgs.use_rcs:
            rcs = copy.deepcopy(points[:, 4])
            rcs_list.append(rcs)

        points = np.column_stack((points[:, :3], np.ones(points.shape[0])))  # Add dummy column if no Doppler

        pointclouds.append(points)
        radar_times.append(time)

    return pointclouds, radar_times, rcs_list, doppler_list, gt_times, gt_positions, gt_quats

def main(cfgs):
    dataset_path = cfgs.data_path
    save_path = os.path.join(cfgs.data_path, 'processed_snail_radar'+ cfgs.radar_topic)
    gt_path = os.path.join(dataset_path, cfgs.GT_folders)
    lidar_calib_path = os.path.join(gt_path, cfgs.lidar_calib_file)
    Body_T_Lidar = load_inhouse_calib(lidar_calib_path)

    for split, group_config in tqdm(cfgs.split_folder.items()):
        save_folder = os.path.join(save_path, split)
        for folder, seq_list in group_config.items():
            if cfgs.radar_topic == '/ars548':
                radar_calib_path = os.path.join(gt_path, folder, cfgs.radar_calib_file)
            else:
                radar_calib_path = os.path.join(gt_path, cfgs.radar_calib_file)

            Body_T_Radar = load_inhouse_calib(radar_calib_path)
            L_T_R = np.linalg.inv(Body_T_Lidar) @ Body_T_Radar

            for seq_num in seq_list:
                seq_name = f"{folder}/{seq_num}"
                place_setup =[k for k, v in cfgs.place_setup_dict.items() if seq_name in v][0]

                save_dir = os.path.join(save_path, save_folder, place_setup, f"{folder}_{seq_num}")
                os.makedirs(save_dir, exist_ok=True)

                seq_path = os.path.join(dataset_path, folder, f"data{seq_num}.bag")
                gt_file = os.path.join(dataset_path, cfgs.GT_folders, folder, f"data{seq_num}", "utm50r_T_xt32.txt")

                pointclouds, radar_times, rcs_list, doppler_list, gt_times, gt_positions, gt_quats = process_bag_data(
                    seq_path, gt_file, cfgs)
                interp_poses = interpolation(np.array(gt_times), np.array(gt_positions), np.array(gt_quats),
                                             np.array(radar_times))

                submap_poses = []
                submap_timestamps = []
                yaws = []
                count = 0

                for i in tqdm(range(0, len(pointclouds), cfgs.gap_size)):
                    end = i + cfgs.frame_winsize
                    if end >= len(pointclouds):
                        continue

                    submap_pc = np.empty((0, 0), dtype=float, order='C')
                    submap_time = int(radar_times[i + cfgs.frame_winsize // 2] * 1e6)
                    center_pose = interp_poses[i + cfgs.frame_winsize // 2]

                    for j in range(i, end):
                        temp_pc = pointclouds[j]
                        Rc_T_U = np.linalg.inv(center_pose @ L_T_R)
                        U_T_Rj = interp_poses[j] @ L_T_R
                        Rc_T_Rj = np.matmul(Rc_T_U, U_T_Rj)
                        temp_pc_in_center = np.matmul(Rc_T_Rj, temp_pc.T).T

                        if cfgs.use_doppler:
                            temp_pc_in_center[:, 3] =  doppler_list[j]

                        if cfgs.use_rcs:
                            temp_pc_in_center = np.column_stack((temp_pc_in_center, rcs_list[j]))

                        submap_pc = np.concatenate((submap_pc, temp_pc_in_center),
                                                   axis=0) if submap_pc.size else temp_pc_in_center

                    # Masking and cleaning
                    mask = np.linalg.norm(submap_pc[:, :3], axis=1) <= cfgs.max_range
                    mask &= submap_pc[:, 2] >= -10
                    mask &= submap_pc[:, 0] >= -10
                    submap_pc = submap_pc[mask]

                    # Downsampling if needed
                    if cfgs.downsample:
                        if len(submap_pc) < cfgs.target_points:
                            additional_points = cfgs.target_points - len(submap_pc)
                            sampled_points = submap_pc[np.random.choice(submap_pc.shape[0], additional_points), :]
                            target_submap = np.concatenate((submap_pc, sampled_points), axis=0)
                        else:
                            target_submap = random_down_sample(submap_pc, cfgs.target_points)

                    else:
                        target_submap = submap_pc

                    submap_name = f"{count:06d}.bin"
                    with open(os.path.join(save_dir, submap_name), 'wb') as f:
                        target_submap.tofile(f)

                    # Save submap and pose
                    submap_pose = np.matmul(center_pose, L_T_R)
                    yaw = Rotation.from_matrix(submap_pose[:3, :3]).as_euler('xyz', degrees=True)[2]
                    submap_poses.append(submap_pose)
                    yaws.append(yaw)
                    submap_timestamps.append(submap_time)
                    count += 1

                submap_poses_path = save_dir + '_poses.txt'
                with open(submap_poses_path, 'w', encoding='utf-8') as f:
                    for pose_id, pose in enumerate(submap_poses):
                        pose_reshape = pose[:3, :4].reshape(1, 12).flatten()
                        time_i = [str(submap_timestamps[pose_id])]
                        yaw_i = np.array([yaws[pose_id]])
                        pose_with_yaw = np.concatenate((time_i, pose_reshape, yaw_i))
                        f.write(' '.join(str(x) for x in pose_with_yaw) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/media/cyw/KESU/datasets/clean_radar/',
                        help='Radar datasets path')
    parser.add_argument('--GT_folders', type=str, default='full_trajs', help='Radar data folder')
    parser.add_argument('--radar_calib_file', type=str, default='Body_T_ars.txt', help='Calibration file path:Body_T_Oculii Body_T_ars.txt')
    parser.add_argument('--lidar_calib_file', type=str, default='Body_T_xt32.txt', help='Calibration folder')
    parser.add_argument('--place_setup_dict', type=list, default=PLACE_SETUP_DICT,
                        help='Groups name in the dataset')
    parser.add_argument('--split_folder', type=dict, default=SPLIT_CONFIG,
                        help='Train, val, and test data split configurations')
    parser.add_argument('--use_rcs', type=bool, default=True, help='Use RCS or not')
    parser.add_argument('--use_doppler', type=bool, default=True, help='Use Doppler velocity or not')
    parser.add_argument('--radar_topic', type=str, default='/ars548', help='Radar topic in rosbag:/radar_enhanced_pcl2 ')
    parser.add_argument('--frame_winsize', type=int, default=7, help='Window size for submap')
    parser.add_argument('--gap_size', type=int, default=5, help='Gap size between submaps')
    parser.add_argument('--downsample', type=bool, default=True, help='Downsample the submap')
    parser.add_argument('--target_points', type=int, default=1024, help='Target points in saved submaps')
    parser.add_argument('--max_range', type=float, default=250, help='Maximum range for the points')

    cfgs = parser.parse_args()
    main(cfgs)
