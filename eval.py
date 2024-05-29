#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import imageio
import cv2
from scene import Scene
import os
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm
from os import makedirs

from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import os
import sys
import numpy as np

import math

from utils.graphics_utils import BasicPointCloud
from waymo_loader import load_waymo_data, load_street_waymo_data
from viewer.camera import CameraOffset

def realtime_rendering(gaussians, model_params, background, scene, dirNum, fixed=False):        
    if args.mode == 1:
            points = model_params[1].cpu()
            colors = np.random.rand(points.shape[0], 3) #[0, 1)
            pcd = BasicPointCloud(points, colors, normals=np.zeros((points.shape[0], 3)))
            scene.gaussians.create_from_pcd(pcd, scene.cameras_extent, scale_ratio=0.1)

    if args.mode in [1, 2, 3, 4]:
        from viewer import viewer
        from PyQt5.QtWidgets import QApplication, QMainWindow
        app = QApplication(sys.argv)
        window = viewer.MainWindow(gaussians, pipeline, background, scene.getTrainCameras(), render, dirNum, mode=args.mode, fixed=fixed)
        return app, window, scene.getTrainCameras()[0][1]
    
def record_rendering(gaussians, pipeline, background, scene, dirNum):
    def setView(view):
        from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixCenterShift
        if cameraOffset is not None:
            novel_view = view
            novel_view.R = (cameraOffset.R @ torch.from_numpy(view.R.T).float()).T
            novel_view.camera_center += cameraOffset.camera_center
            novel_view.T = - novel_view.R.transpose(0,1) @ novel_view.camera_center
            Rt = torch.zeros((4, 4))
            Rt[:3, :3] = novel_view.R.transpose(0,1)
            Rt[:3, 3] = novel_view.T
            Rt[3, 3] = 1.0
            novel_view.world_view_transform = Rt.transpose(0, 1)

            focal_x = math.tan(view.FoVx / 2.0) / 2.0 * view.image_width
            focal_y = math.tan(view.FoVy / 2.0) / 2.0 * view.image_height
            ratio = focal_y / focal_x
            focal_x += cameraOffset.focal_x
            focal_y += cameraOffset.focal_x * ratio

            novel_view.image_height = cameraOffset.height
            novel_view.image_width = cameraOffset.width
            novel_view.FoVx = math.atan(focal_x / novel_view.image_width * 2.0) * 2.0
            novel_view.FoVy = math.atan(focal_y / novel_view.image_height * 2.0) * 2.0

            if novel_view.cx > 0:
                novel_view.projection_matrix = getProjectionMatrixCenterShift(novel_view.znear, novel_view.zfar, novel_view.cx, novel_view.cy, novel_view.fl_x, novel_view.fl_y, novel_view.image_width, novel_view.image_height).transpose(0,1)
            else:
                novel_view.projection_matrix = getProjectionMatrix(znear=novel_view.znear, zfar=novel_view.zfar, fovX=novel_view.FoVx, fovY=novel_view.FoVy).transpose(0,1)
            novel_view.full_proj_transform = (novel_view.world_view_transform.unsqueeze(0).bmm(novel_view.projection_matrix.unsqueeze(0))).squeeze(0) # full projection transform
            
        return novel_view.cuda()


    view_path = "view.obj"
    cameraOffset = None
    if os.path.exists(view_path):
        cameraOffset = torch.load(view_path)

    viewDir = args.viewDir
    assert 0 <= viewDir < dirNum
    cameraSet = scene.getTrainCameras()
    frames = []
    if args.mode == 5:
        for frameCnt in range(0, len(cameraSet), dirNum):
            rendered = render(setView(cameraSet[frameCnt + viewDir][1]), gaussians, pipeline, background)["render"]
            rendered = torch.clamp(rendered, 0, 1)
            rendered = (rendered*255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            image = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
            frames.append(image)
        imageio.mimsave("render_video.mp4", np.array(frames), fps=10)
    elif args.mode == 6:
        for frameCnt in range(0, len(cameraSet), dirNum):
            rendered = render(setView(cameraSet[frameCnt + viewDir][1]), gaussians, pipeline, background)["depth"].detach()[0]
            rendered = rendered.cpu().numpy() 
            normalized_depth = (rendered - np.min(rendered)) / (np.max(rendered) - np.min(rendered))
            image = cv2.applyColorMap(np.uint8(normalized_depth * 255), cv2.COLORMAP_JET)
            frames.append(image)
        imageio.mimsave("depth_video.mp4", np.array(frames), fps=10)

def eval_metrics(gaussians, pipeline, background, scene, dirNum):
    from utils.image_utils import psnr
    from utils.loss_utils import ssim
    from lpipsPyTorch import lpips
    train_metrics = {"psnr": [], "ssim": [], "lpips": []}
    test_metrics = {"psnr": [], "ssim": [], "lpips": []}
    cameraSet = scene.getTrainCameras()

    for frameCnt in range(0, len(cameraSet), dirNum):
        for viewDir in range(dirNum):
            gt, cam = cameraSet[frameCnt + viewDir]
            rendered = render(cam.cuda(), gaussians, pipeline, background)["render"]
            rendered = rendered.cpu()
            gt = gt / 255.0
            if cam.gt_alpha_mask is not None:
                gt *= 1 - cam.gt_alpha_mask / 255.0
            if (frameCnt // dirNum + 1) % 10 == 0:
                test_metrics["psnr"].append(psnr(rendered, gt))
                test_metrics["ssim"].append(ssim(rendered, gt))
                test_metrics["lpips"].append(lpips(rendered, gt))
            else: 
                train_metrics["psnr"].append(psnr(rendered, gt))
                train_metrics["ssim"].append(ssim(rendered, gt))
                train_metrics["lpips"].append(lpips(rendered, gt))
    data = [
        ["", "PSNR", "SSIM", "LPIPS"],
        ["Train Set", np.mean(train_metrics["psnr"]), np.mean(train_metrics["ssim"]), np.mean(train_metrics["lpips"])],
        ["Test Set", np.mean(test_metrics["psnr"]), np.mean(test_metrics["ssim"]), np.mean(test_metrics["lpips"])]
    ]

    column_widths = [10, 6, 6, 6]
    header = data[0]
    rows = data[1:]
    print("{:<{}} {:>{}} {:>{}} {:>{}}".format(header[0], column_widths[0], header[1], column_widths[1], header[2], column_widths[2], header[3], column_widths[3]))
    print("-" * (sum(column_widths) + len(column_widths) - 1))

    for row in rows:
        print("{:<{}} {:>0.4f} {:>0.4f} {:>0.4f}".format(row[0], column_widths[0], float(row[1]), float(row[2]), float(row[3])))
            

def distributeTask(dataset : ModelParams, pipeline : PipelineParams):
    if args.mode in [2, 3, 5, 6, 7]:
        fixed = True
    elif args.mode in [1, 4]:
        fixed = False
    elif args.mode == 0:
        raise ValueError("Error: mode not assigned")
    else:
        raise ValueError("Error: invalid mode")
    
    dirNum = args.camera_number
    
    with torch.no_grad():
        if "waymo" in args.source_path:
            gaussians, scene = load_waymo_data(args.source_path, args, test=True)
        else:
            gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=args.gaussian_dim, time_duration=args.time_duration, rot_4d=args.rot_4d, force_sh_3d=args.force_sh_3d, sh_degree_t=2 if pipeline.eval_shfs_4d else 0)
            scene = Scene(dataset, gaussians, shuffle=False)
        (model_params, first_iter) = torch.load(args.pth)
        print("iter: ", first_iter)
        gaussians.restore(model_params, args)
        # model_params = None

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if args.mode in [1, 2, 3, 4]:
            return realtime_rendering(gaussians, model_params, background, scene, dirNum, fixed)
        elif args.mode in [5, 6]:
            record_rendering(gaussians, pipeline, background, scene, dirNum)
        elif args.mode in [7]:
            eval_metrics(gaussians, pipeline, background, scene, dirNum)
        
    
# python render.py --config configs/dnerf/standup.yaml --pth output/dnerf/standup/chkpnt30000.pth --skip_test
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--config", type=str)
    parser.add_argument("--pth", type=str)
    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--viewDir", type=int, default=0)
    parser.add_argument("--frame_length", type=int, default=[0, 50])
    parser.add_argument("--camera_number", type=int, default=1)
    parser.add_argument("--use_skymask", type=bool, default=False)

    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    args.source_path = ""
    args.model_path = ""
    args.time_duration = args.rot_4d = args.force_sh_3d = args.gaussian_dim = None
    
    cfg = OmegaConf.load(args.config)
    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                recursive_merge(key1, host[key])
        else:
            setattr(args, key, host[key])
    for k in cfg.keys():
        recursive_merge(k, cfg)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if args.mode not in [5, 6, 7]:
        app, window, org_view = distributeTask(model.extract(args), pipeline.extract(args))
        window.show()
        app.exec_()
        view = window.renderScene.view
        camOffset = CameraOffset(
            R = view.R.T.cpu() @ org_view.R.T.inverse(),
            camera_center = view.camera_center.cpu() - org_view.camera_center,
            focal_x = math.tan(view.FoVx / 2.0) / 2.0 * view.image_width - math.tan(org_view.FoVx / 2.0) / 2.0 * org_view.image_width,
            height = view.image_height,
            width = view.image_width
        )
        torch.save(camOffset, "view.obj")
        sys.exit()
    else :
        distributeTask(model.extract(args), pipeline.extract(args))
    