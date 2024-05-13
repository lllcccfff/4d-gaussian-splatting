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

from utils.graphics_utils import BasicPointCloud
from waymo_loader import load_waymo_data, load_street_waymo_data

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
        window = viewer.MainWindow(gaussians, pipeline, background, scene.getTrainCameras(), render, args.mode, dirNum, fixed=fixed)
        return app, window
    
def record_rendering(gaussians, pipeline, background, scene, dirNum):
    viewDir = 1
    assert 0 <= viewDir < dirNum
    cameraSet = scene.getTrainCameras()
    frames = []
    if args.mode == 5:
        for frameCnt in range(0, len(cameraSet), dirNum):
            rendered = render(cameraSet[frameCnt + viewDir][1].cuda(), gaussians, pipeline, background)["render"]
            rendered = (rendered*255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            image = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
            frames.append(image)
        imageio.mimsave("render_video.mp4", np.array(frames), fps=10)
    elif args.mode == 6:
        for frameCnt in range(0, len(cameraSet), dirNum):
            rendered = render(cameraSet[frameCnt + viewDir][1].cuda(), gaussians, pipeline, background)["depth"].detach()[0]
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
            if frameCnt % 10 == 0:
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
            

def distributeTask(dataset : ModelParams, pipeline : PipelineParams, args, dirNum=3):
    if args.mode in [2, 3, 5, 6, 7]:
        fixed = True
    elif args.mode in [1, 4]:
        fixed = False
    elif args.mode == 0:
        raise ValueError("Error: mode not assigned")
    else:
        raise ValueError("Error: invalid mode")
    
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
        app, window = distributeTask(model.extract(args), pipeline.extract(args), args)
        window.show()
        sys.exit(app.exec_())
    else :
        distributeTask(model.extract(args), pipeline.extract(args), args)
    