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
    with torch.no_grad():
        if "waymo" in args.source_path:
            gaussians, scene = load_waymo_data(args.source_path, args, test=True)
        else:
            gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=args.gaussian_dim, time_duration=args.time_duration, rot_4d=args.rot_4d, force_sh_3d=args.force_sh_3d, sh_degree_t=2 if pipeline.eval_shfs_4d else 0)
            scene = Scene(dataset, gaussians, shuffle=False)
        (model_params, first_iter) = torch.load(args.pth)
        gaussians.restore(model_params, args)
        # points = model_params[1].cpu()
        # colors = np.random.rand(points.shape[0], 3) #[0, 1)
        # pcd = BasicPointCloud(points, colors, normals=np.zeros((points.shape[0], 3)))
        # scene.gaussians.create_from_pcd(pcd, scene.cameras_extent)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


        app = QApplication(sys.argv)
        window = viewer.MainWindow(gaussians, pipeline, background, scene.getTrainCameras(), render, fixed=True)
        return app, window
    
# python render.py --config configs/dnerf/standup.yaml --pth output/dnerf/standup/chkpnt30000.pth --skip_test
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--config", type=str)
    parser.add_argument("--pth", type=str)
    # parser.add_argument("--skip_train", action="store_true")
    # parser.add_argument("--skip_test", action="store_true")
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

    app, window = launch(model.extract(args), pipeline.extract(args), args)
    print(1)
    window.show()
    sys.exit(app.exec_())
    