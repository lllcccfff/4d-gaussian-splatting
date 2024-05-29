import torch 
class CameraOffset:
    R : torch.Tensor
    camera_center : torch.Tensor
    focal_x : float
    height : int
    width : int
    def __init__(self, R, camera_center, focal_x, height, width):
        self.R = R
        self.camera_center = camera_center
        self.focal_x = focal_x
        self.height = height
        self.width = width