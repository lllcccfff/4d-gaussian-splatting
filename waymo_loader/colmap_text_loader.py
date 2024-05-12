import os, json
import torch, numpy


class Image:
    qvec : torch.Tensor #w2c
    tvec : torch.Tensor #w2c
    def __init__(self, qvec, tvec) -> None:
        self.qvec, self.tvec = torch.tensor(qvec), torch.tensor(tvec)

def qvec2mat(q):
    return 2 * torch.stack([
        torch.stack([.5 - q[..., 2]**2 - q[..., 3]**2, q[..., 1]*q[..., 2] - q[..., 0]*q[..., 3], q[..., 3]*q[..., 1] + q[..., 0]*q[..., 2]], dim=-1),
        torch.stack([q[..., 1]*q[..., 2] + q[..., 0]*q[..., 3], .5 - q[..., 1]**2 - q[..., 3]**2, q[..., 2]*q[..., 3] - q[..., 0]*q[..., 1]], dim=-1),
        torch.stack([q[..., 3]*q[..., 1] - q[..., 0]*q[..., 2], q[..., 2]*q[..., 3] + q[..., 0]*q[..., 1], .5 - q[..., 1]**2 - q[..., 2]**2], dim=-1)
    ], dim=-2)

def get_viewmatrix(img : Image):
    R = qvec2mat(img.qvec)
    rt = torch.cat([R, img.tvec.unsqueeze(-1)], dim=-1)
    return torch.cat([rt, torch.tensor([[0,0,0,1.]])])


def read_points3D_text(basedir):
    xyzs, rgbs = [], []
    path = os.path.join(basedir, "sparse/0/points3D.txt")
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyzs.append(torch.tensor(tuple(map(float, elems[1:4]))))
                rgbs.append(torch.tensor(tuple(map(int, elems[4:7]))))
    return torch.stack(xyzs, dim=0), torch.stack(rgbs, dim=0)

def read_w2c(basedir):    
    path = os.path.join(basedir, "sparse/0/images.txt")
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image = Image(tuple(map(float, elems[1:5])), tuple(map(float, elems[5:8])))
                return image
            
def get_spatial_scale(basedir):    
    path = os.path.join(basedir, "sparse/0/images.txt")
    img_basedir = os.path.join(basedir, "images")
    with open(path, "r") as f:
        lines = f.readlines()
        images = []
        for line in lines:
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image = Image(tuple(map(float, elems[1:5])), tuple(map(float, elems[5:8])))
                viewmatrix = get_viewmatrix(image)
                image.camera_center = viewmatrix.inverse()[:3, 3]
                images.append(image)

    camera_pos=[]
    for image in images:
        camera_pos.append(image.camera_center)
    camera_pos = torch.stack(camera_pos)
    camera_center = torch.mean(camera_pos, dim=0)
    dist = torch.sum((camera_pos - camera_center) ** 2, dim=-1)
    max_dist = torch.max(dist)
    return torch.sqrt(max_dist) * 1.1

def load_data(basedir):
    xyzs, rgbs = read_points3D_text(basedir)
    image = read_w2c(basedir)
    w_2c = get_viewmatrix(image)

    # spatial_scale = get_spatial_scale(basedir)
    return xyzs, rgbs, w_2c
