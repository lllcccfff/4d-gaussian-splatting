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
from viewer import viewer
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
import numpy as np

from utils.graphics_utils import BasicPointCloud
from waymo_loader import load_waymo_data

def launch(dataset : ModelParams, pipeline : PipelineParams, args):
    if args.mode in [2, 3, 5]:
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
        gaussians.restore(model_params, args)

        if args.mode == 1:
            points = model_params[1].cpu()
            colors = np.random.rand(points.shape[0], 3) #[0, 1)
            pcd = BasicPointCloud(points, colors, normals=np.zeros((points.shape[0], 3)))
            scene.gaussians.create_from_pcd(pcd, scene.cameras_extent, scale_ratio=0.1)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if args.mode in [1, 2, 3, 4]:
            app = QApplication(sys.argv)
            window = viewer.MainWindow(gaussians, pipeline, background, scene.getTrainCameras(), render, args.mode, fixed=fixed)
            return app, window
        elif args.mode == 5:
            viewDir = 0
            cameraSet = scene.getTrainCameras()
            frames = []
            for frameCnt in range(0, 995, 5):
                rendered = render(cameraSet[frameCnt][1].cuda(), gaussians, pipeline, background)["render"]
                rendered = (rendered*255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                image = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
                frames.append(image)
            imageio.mimsave("video.mp4", np.array(frames), fps=10)
    
# python render.py --config configs/dnerf/standup.yaml --pth output/dnerf/standup/chkpnt30000.pth --skip_test
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--config", type=str)
    parser.add_argument("--pth", type=str)
    #point cloud/depth map/fix camera and trace/free camera
    # 0: nto assigned error, 1: point cloud, 2: depth map, 3: fix camera and trace, 4: free camera, 5: output fix camera trace
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

    if args.mode != 5:
        app, window = launch(model.extract(args), pipeline.extract(args), args)
        print(1)
        window.show()
        sys.exit(app.exec_())
    else :
        launch(model.extract(args), pipeline.extract(args), args)
    