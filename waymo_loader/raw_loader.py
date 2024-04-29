import numpy as np
import os
import torch
import cv2

def load_ego_pose(ego_pose_dir):
    ego_pose = []
    for filename in sorted(os.listdir(ego_pose_dir)):
            fp = os.path.join(ego_pose_dir, filename)
            matrix = np.loadtxt(fp)
            ego_pose.append(matrix)
    return ego_pose #ego to world

def load_lidar(lidar_dir):
    lidar = []
    
    for filename in sorted(os.listdir(lidar_dir)):
        fp = os.path.join(lidar_dir, filename)
        point_cloud = np.fromfile(fp, dtype=np.float32)
        point_cloud = point_cloud.reshape((-1, 14))  # 每行14个元素
        origins = point_cloud[:, 0:3]  # 起点坐标 [N, 3]
        points = point_cloud[:, 3:6]  # 点坐标 [N, 3]
        flows = point_cloud[:, 6:10]  # 流信息 [N, 4]
        ground_label = point_cloud[:, 10]  # 地面标签 [N], 原始为bool, 保存为float
        intensity = point_cloud[:, 11]  # 强度 [N]
        elongation = point_cloud[:, 12]  # 伸展度 [N]
        laser_ids = point_cloud[:, 13]  # 激光ID [N]
        lidar.append({
            "origins": origins,
            "points": points,
            "flows": flows,
            "ground_label": ground_label,
            "intensity": intensity,
            "elongation": elongation,
            "laser_ids": laser_ids
        })

    return lidar

def load_image(images_dir):
    files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') and "_" in f]
    num_groups = len(set(f.split('_')[0] for f in files))
    num_variants = len([f for f in files if f.startswith('000_')])
    np_images = np.empty((num_groups, num_variants), dtype=object)
    images_pth = []

    for file in files:
        fp = os.path.join(images_dir, file)
        img = cv2.imread(fp) # h, w, c
        group_index, variant_index = map(int, file.split('.')[0].split('_'))
        np_images[group_index, variant_index] = img
        images_pth.append(file)

    images = [np_images[i] for i in range(np_images.shape[0])]
    return images, images_pth

def load_intrinsics(intrinsics_dir):
    intrinsics = []
    for filename in sorted(os.listdir(intrinsics_dir)):
            fp = os.path.join(intrinsics_dir, filename)
            matrix = np.loadtxt(fp)
            intrinsics.append(matrix)
    return intrinsics #ego to world

def load_extrinsics(extrinsics_dir):
    extrinsics = []
    for filename in sorted(os.listdir(extrinsics_dir)):
            fp = os.path.join(extrinsics_dir, filename)
            matrix = np.loadtxt(fp)
            extrinsics.append(matrix)
    return extrinsics #ego to world

def load_waymo_raw(base_dir):
    ego_pose = load_ego_pose(os.path.join(base_dir, "ego_pose"))
    print("[Loaded] ego_pose")
    lidar = load_lidar(os.path.join(base_dir, "lidar"))
    print("[Loaded] lidar")
    images, img_pth = load_image(os.path.join(base_dir, "images"))
    print("[Loaded] images")
    intrinsics = load_intrinsics(os.path.join(base_dir, "intrinsics"))
    print("[Loaded] intrinsics")
    extrinsics = load_extrinsics(os.path.join(base_dir, "extrinsics"))
    print("[Loaded] extrinsics")

    return {
        "ego_pose": ego_pose,
        "lidar": lidar,
        "images": images,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "otthers": None
    }
    