import numpy as np
import os
import torch
import random
import cv2
import tracemalloc
from PIL import Image
from arguments import ModelParams
from scene import Scene, GaussianModel, dataset_readers
from utils.graphics_utils import BasicPointCloud
from utils.camera_utils import cameraList_from_camInfos

class SceneWaymo(Scene):
    def __init__(self, gaussians: GaussianModel, args, waymo_raw_pkg, shuffle=True, resize_ratio=1, test=False):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.white_background = args.white_background
        self.train_cameras = {}
        self.test_cameras = {}
        self.cameras_extent = None

        ego_pose, lidar, images, intrinsics, extrinsics, sky_masks = waymo_raw_pkg["ego_pose"], waymo_raw_pkg["lidar"], waymo_raw_pkg["images"], waymo_raw_pkg["intrinsics"], waymo_raw_pkg["extrinsics"], waymo_raw_pkg["sky_mask"]
        frame_num = args.frame_length
        train_cameras_raw = []
        test_camera_raw = []
        opencv2waymo = np.array([[0, 0, 1, 0], 
                                 [-1, 0, 0, 0], 
                                 [0, -1, 0, 0], 
                                 [0, 0, 0, 1]])

        for i in range(frame_num[0], frame_num[1]):
            timestamp = (args.time_duration[1] - args.time_duration[0]) / (frame_num[1] - frame_num[0]) * i + args.time_duration[0]
            for j in range(args.camera_number):
                # --------------undistort----------------
                # distorted_img = images[i][j]
                # cameraMatrix = np.array([[fx, 0, cx],
                #                          [0, fy, cy],
                #                          [0,  0,  1]])
                # distCoeffs = np.array([k1, k2, p1, p2, k3])
                # h, w = distorted_img.shape[:2]
                # newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 1, (w, h))
                # undistorted_img = cv2.undistort(distorted_img, cameraMatrix, distCoeffs, None, newCameraMatrix)
                # x, y, w, h = roi
                # fx, fy = newCameraMatrix[0, 0], newCameraMatrix[1, 1]
                # undistorted_img = undistorted_img[y:y+h, x:x+w]
                # #norm_ud_img =  (undistorted_img / 255.0).astype(np.float32)
                # pil_img = Image.fromarray(undistorted_img)
                # --------------distort----------------
                if args.use_skymask:
                    pil_img = Image.fromarray(np.concatenate([images[i][j], sky_masks[i][j][..., 0:1]], axis=2))
                else:
                    pil_img = Image.fromarray(images[i][j])
                width, height = pil_img.size
                fl_x, fl_y, cx, cy, k1, k2, p1, p2, k3 = intrinsics[j]

                if resize_ratio != 1:
                    pil_img = pil_img.resize((int(width * resize_ratio), int(height * resize_ratio)), Image.Resampling.LANCZOS)
                    width, height = pil_img.size
                    fl_x, fl_y, cx, cy = fl_x * resize_ratio, fl_y * resize_ratio, cx * resize_ratio, cy * resize_ratio
                FovX, FovY = 2 * np.arctan(width / (2 * fl_x)), 2 * np.arctan(height / (2 * fl_y))

                c2w = ego_pose[i] @ extrinsics[j] @ opencv2waymo # ego2world * waymo2ego * opencv2waymo
                w2c = np.linalg.inv(c2w)
                R = w2c[:3, :3].copy()
                T = w2c[:3, 3].copy()

                camera = dataset_readers.CameraInfo(
                    uid=None, R=R.T, T=T, FovY=FovY, FovX=FovX, image=pil_img, depth=None,
                    image_path=None, image_name=None, width=width, height=height, timestamp=timestamp)
                
                if (i+1) % 10 == 0 and test is False:
                    test_camera_raw.append(camera)
                    train_cameras_raw.append(camera)
                else :
                    train_cameras_raw.append(camera)

        nerf_normalization = dataset_readers.getNerfppNorm(train_cameras_raw)
        if shuffle and test is False:
            random.shuffle(train_cameras_raw)
            random.shuffle(test_camera_raw)

        self.cameras_extent = 200
        self.train_cameras[args.resolution] = cameraList_from_camInfos(train_cameras_raw, args.resolution, args)
        self.test_cameras[args.resolution] = cameraList_from_camInfos(test_camera_raw, args.resolution, args)
        print("[Loaded] cameras")

        #depth map
        if not test and args.supervise_depth:
        # if not test and args.supervise_depth:
            # all_frame_points = []
            # for frame in range(len(images)): # debug
            #     homo_points = np.concatenate([lidar[frame]['points'], np.ones((lidar[frame]['points'].shape[0], 1))], axis=1)
            #     homo_points = homo_points @ ego_pose[frame].T
            #     points = homo_points / homo_points[:, 3, None]
            #     all_frame_points.append(points)
            # points = np.concatenate(all_frame_points) # points are in world coordination

            # del all_frame_points
            # points = torch.from_numpy(points).cuda()
            # mask = np.random.randint(0, points.shape[0], points.shape[0]//50)
            # points = points[mask]

            for i in range(len(self.train_cameras[args.resolution])):
                camera = self.train_cameras[args.resolution][i]
                frame = frame_num[0] + i // args.camera_number
                homo_points = np.concatenate([lidar[frame]['points'], np.ones((lidar[frame]['points'].shape[0], 1))], axis=1)
                homo_points = homo_points @ ego_pose[frame].T
                points = homo_points / homo_points[:, 3, None]
                points = torch.from_numpy(points).cuda()

                h, w = camera.image_height, camera.image_width
                points_camera = points.float() @ camera.full_proj_transform.cuda()
                points_camera = points_camera[:, :3] / points_camera[:, 3, None]
                points_camera[:, 0] = ((points_camera[:, 0] + 1.0) * w - 1) * 0.5
                points_camera[:, 1] = ((points_camera[:, 1] + 1.0) * h - 1) * 0.5
                uvz = points_camera

                # PVG 的方式 有bug
                # K = torch.tensor([
                #     [camera.fl_x, 0, camera.cx],
                #     [0, camera.fl_y, camera.cy],
                #     [0, 0, 1]
                # ], dtype=torch.float32, device="cuda")
                # points_camera = points.float() @ camera.world_view_transform.cuda()
                # points_camera = points_camera[:, :3] / points_camera[:, 3, None]
                # uvz = points_camera @ K.T
                # uvz[:, :2] /= uvz[:, 2:]

                uvz = uvz[uvz[:, 2] > 0]
                uvz = uvz[uvz[:, 1] >= 0]
                uvz = uvz[uvz[:, 1] < h]
                uvz = uvz[uvz[:, 0] >= 0]
                uvz = uvz[uvz[:, 0] < w]
                uv = uvz[:, :2].to(torch.int32)

                # deep test
                pts_depth = torch.zeros([1, h, w], device="cuda") 
                pts_depth[0, uv[:, 1], uv[:, 0]] = uvz[:, 2]
                # uv = uvz[:, :2].to(torch.int32).cpu()
                # depths = uvz[:, 2].cpu()
                # for i in range(len(uv)):
                #     u = uv[i, 0]
                #     v = uv[i, 1]
                #     z = depths[i]
                #     if pts_depth[0, v, u] == 0 or z < pts_depth[0, v, u]:
                #         pts_depth[0, v, u] = z
                camera.depth_map = pts_depth.float().cpu()


        #gaussian
        
        # LiDAR/SfM/random
        mode = 2
        if mode == 0:
            all_frame_points = []
            for frame in range(frame_num[0], frame_num[1]):
                homo_points = np.concatenate([lidar[frame]['points'], np.ones((lidar[frame]['points'].shape[0], 1))], axis=1)
                homo_points = homo_points @ ego_pose[frame].T
                points = homo_points[:, :3] / homo_points[:, 3, None]
                all_frame_points.append(points)
            points = np.concatenate(all_frame_points)
            mask = np.random.randint(0, points.shape[0], points.shape[0]//(frame_num[1] - frame_num[0]))
            points = points[mask]
            colors = np.random.rand(points.shape[0], 3) #[0, 1)a
        elif mode == 1:        
            SfM_clouds = waymo_raw_pkg["SfM_clouds"]
            points = SfM_clouds[0]
            colors = SfM_clouds[1] / 255.
        elif mode == 2:
            points_num  = 50_000
            cam_centers = []
            for cam in self.train_cameras[args.resolution]:
                cam_centers.append(cam.camera_center)
            cam_centers = torch.stack(cam_centers)
            avg_center = cam_centers.mean(dim=0).numpy()
            points = np.zeros((points_num, 3 ))
            for i in range(3):
                points[:, i] = np.random.uniform(avg_center[i] - self.cameras_extent * 0.1, avg_center[i] + self.cameras_extent * 0.1, size=points_num)
            colors = np.random.rand(points_num, 3) #[0, 1)

        if points.shape[0] > args.num_pts:
            mask = np.random.randint(0, points.shape[0], args.num_pts)
            points = points[mask]
        pcd = BasicPointCloud(points, colors, normals=np.zeros((points.shape[0], 3)))
        self.gaussians.create_from_pcd(pcd, self.cameras_extent)
        print("[Loaded] guassians")

def load_waymo_gs(waymo_raw_pkg, args, test):
    gaussians = GaussianModel(args.sh_degree, gaussian_dim=args.gaussian_dim, time_duration=args.time_duration, rot_4d=args.rot_4d, force_sh_3d=args.force_sh_3d, sh_degree_t=2 if args.eval_shfs_4d else 0)
    scene = SceneWaymo(gaussians, args, waymo_raw_pkg, test=test)
    return gaussians, scene