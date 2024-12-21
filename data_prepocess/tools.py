# -*-coding:utf-8-*-
import numpy as np
import random
from scipy.spatial.transform import Rotation
from test_slerp import pos_interpolate_batch, rot_slerp_batch

def find_nearest_timestamp(timestamp, timestamps):
    idx = np.argmin(np.abs(timestamps - timestamp))
    return timestamps[idx]

def interpolation(gt_times, gt_positions, gt_quats, radar_times):
    interp_positions = pos_interpolate_batch(gt_times, gt_positions, radar_times)
    interp_quats = rot_slerp_batch(gt_times, gt_quats, radar_times)
    poses = []
    for i, q_i in enumerate(interp_quats):
        quat_max = Rotation.from_quat(q_i).as_matrix()
        xyz = interp_positions[i].reshape(3, 1)
        pose = np.hstack((quat_max, xyz))
        pose = np.vstack((pose, [0, 0, 0, 1]))
        poses.append(pose)
    return poses

def random_down_sample(pc, sample_points):
    sampleA = random.sample(range(pc.shape[0]), sample_points)
    sampled_pc = pc[sampleA]
    return sampled_pc

def load_poses(poses_path, sign):
    positions = []
    quats = []
    gt_times = []
    with open(poses_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            data = line.strip().split(sign)
            if (data[0].startswith('#')):
                continue
            # if len(data) >= 8:  # 确保每行有8个数据项
            #     time = float(data[0])
            #     tx, ty, tz, qx, qy, qz, qw = map(float, data[1:8])
            #     gt_times.append(time)
            #     xyz.append([tx, ty, tz, qx, qy, qz, qw])

            time, *data = np.fromstring(line, dtype=np.float64, sep=sign)
            gt_times.append(time)
            xyz = data[:3]
            quaternion = data[3:7]
            positions.append(xyz)
            quats.append(quaternion)
    return gt_times, positions, quats

def load_poses_matrix(poses_path, sign):
    poses=[]
    gt_times = []
    with open(poses_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i == 0 and line.strip().split(sign)[0].startswith('#'):
                continue
            time, *data = np.fromstring(line, dtype=np.float64, sep=sign)
            gt_times.append(time)
            xyz = np.array(data[:3], dtype=np.float64).reshape(3, 1)
            quaternion = np.array(data[3:7], dtype=np.float64)
            quat_max = Rotation.from_quat(quaternion).as_matrix()
            pose = np.hstack((quat_max, xyz))
            pose = np.vstack((pose, [0, 0, 0, 1]))
            poses.append(pose)
    return gt_times, poses

def standardization(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc_normalized = pc / m
    return pc_normalized

def standardization_rcs_255(pci):
    pc = pci[:, :3]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc_normalized = pc / m

    scale = 255.0
    intensity_norm = pci[:, 3] / scale

    pci[:, :3] = pc_normalized
    pci[:, 3] = intensity_norm
    return pci

def standardization_rcs(pci):
    pc = pci[:, :3]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc_normalized = pc / m

    intensity = pci[:, 3]
    scale = np.max(intensity) - np.min(intensity)
    scale = np.maximum(scale, 1e-8)
    intensity_norm = (intensity - np.min(intensity)) / scale

    pci[:, :3] = pc_normalized
    pci[:, 3] = intensity_norm
    return pci

def standardization_pv(pci):
    pc = np.copy(pci[:, :3])
    centroid = np.mean(pc, axis=0)

    # pc1 = pc - centroid
    # m = np.max(np.sqrt(np.sum(pc1 ** 2, axis=1)))
    # pc_normalized = pc1 / m
    pc_diff = pc - centroid
    sum = np.sum(np.sqrt(np.sum(pc_diff ** 2, axis=1)))
    d = sum / len(pc)
    scale = 0.5 / d

    T = np.array([[scale, 0, 0, -scale*centroid[0]],
         [0, scale, 0, -scale*centroid[1]],
         [0, 0, scale, -scale *centroid[2]],
         [0, 0, 0, 1]])
    pc_new = np.hstack((pc, np.ones([len(pc), 1])))
    scaled_output = np.matmul(T, pc_new.T).T

    intensity = pci[:, 3]
    scale = np.max(intensity) - np.min(intensity)
    scale = np.maximum(scale, 1e-8)
    intensity_norm = (intensity - np.min(intensity)) / scale

    pci[:, :3] = scaled_output[:, :3]
    pci[:, 3] = intensity_norm

    pci = pci[(pci[:, 0] >= -1) & (pci[:, 0] <= 1) &
              (pci[:, 1] >= -1) & (pci[:, 1] <= 1) &
              (pci[:, 2] >= -1) & (pci[:, 2] <= 1)]
    return pci

def norm_rcs(pci):
    intensity = pci[:, 3]
    scale = np.max(intensity) - np.min(intensity)
    scale = np.maximum(scale, 1e-8)
    intensity_norm = (intensity - np.min(intensity)) / scale

    pci[:, 3] = intensity_norm

    return pci

def scale_rcs_to_x(pci):
    rcs = pci[:, 3]
    x = pci[:, 0]
    rcs_range = np.max(rcs) - np.min(rcs)
    x_range = np.max(x) - np.min(x)
    scale_factor = x_range/rcs_range
    if scale_factor == 0:
        scale_factor=1e-6
    scale_rcs = rcs-np.min(rcs)*scale_factor + np.min(rcs)

    pci[:, 3] = scale_rcs

    return pci

def rigid_transform_3D(A, B):
    """
    Calculate the rigid transformation matrix between point sets A and B.
    Assumes A and B are both 3D points (rows are points, columns are xyz).
    """
    assert len(A) == len(B)

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    H = np.dot((A - centroid_A).T, B - centroid_B)

    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    t = -np.dot(R, centroid_A) + centroid_B

    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T

def covert_tumpose_to_matrix(position,quats):
    x = position[0]
    y = position[1]
    z = position[2]
    qx = quats[0]
    qy = quats[1]
    qz = quats[2]
    qw = quats[3]
    Rot_mat = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    return np.array([[Rot_mat[0,0], Rot_mat[0,1], Rot_mat[0,2], x],
                     [Rot_mat[1,0], Rot_mat[1,1], Rot_mat[1,2], y],
                     [Rot_mat[2,0], Rot_mat[2,1], Rot_mat[2,2], z],
                     [0, 0, 0, 1]])

def convert_matrix_to_tumpose(mat):
    Rot_mat = mat[0:3, 0:3]
    r = Rotation.from_matrix(Rot_mat)
    quat = r.as_quat()
    return [mat[0,3], mat[1,3], mat[2,3], quat[0], quat[1], quat[2], quat[3]]