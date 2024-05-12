import numpy
import os
import torch
from waymo_loader import raw_loader, gs_loader, street_gs_loader

def load_waymo_data(data_dir, args, test=False):
    print("\n====== [Loading] Waymo Open Dataset ======")
    waymo_raw_pkg = raw_loader.load_waymo_raw(data_dir)
    print("------------")
    gaussians, scene = gs_loader.load_waymo_gs(waymo_raw_pkg, args, test=test)
    print("====== [Successd] Waymo Open Dataset ======\n")
    return gaussians, scene
    
def load_street_waymo_data(data_dir, args, test=False):
    print("\n====== [Loading] Waymo Open Dataset ======")
    waymo_raw_pkg = raw_loader.load_waymo_raw(data_dir)
    print("------------")
    gaussians, scene = street_gs_loader.load_waymo_gs(waymo_raw_pkg, args, test=test)
    print("====== [Successd] Waymo Open Dataset ======\n")
    return gaussians, scene