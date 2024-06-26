import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer, QDateTime
from PyQt5.QtWidgets import QSizePolicy

import torchvision.transforms.functional as TF
import cv2
import torch
import math
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixCenterShift
# interface
# class RenderScene:
#     def __init__(self):
#         raise NotImplementedError
#     def __call__(self, t):
#         raise NotImplementedError
#     @set
#     def setViewPos(self, view):
#         self.view = view
#     def setViewRot(self, view):
#         self.view = view

# class RenderScene4DGS(RenderScene):
class RenderScene4DGS:
    def __init__(self, gaussians, pipeline, background, viewSet, render_fn, dirNum, mode, fixed):
        self.gaussians = gaussians
        self.pipeline = pipeline
        self.background = background
        self.render_fn = render_fn
        self.fixed = fixed
        self.mode = mode
        self.dirNum = dirNum
        
        self.frameCnt = 0

        self.viewDir = 0
        self.viewSet = viewSet
        init_view = self.viewSet[0][1]
        init_view.R = torch.from_numpy(init_view.R).float()
        init_view.T = torch.from_numpy(init_view.T).float()
        init_view = init_view.cuda()
        self.view = init_view

        self.width = self.view.image_width
        self.height = self.view.image_height

    def decimal_to_256(self, decimal):
        result = []
        flag=3
        while flag > 0:
            flag -=1
            remainder = decimal % 256
            result.append(remainder)
            decimal //= 256
        return result[::-1] 
    
    def __call__(self, t):
        time_duration = self.gaussians.time_duration[1] - self.gaussians.time_duration[0]
        if self.fixed:
            self.frameCnt %= len(self.viewSet)
            self.setView(self.frameCnt + self.viewDir)
            self.frameCnt += self.dirNum
        else:
            self.view.timestamp = time_duration * ((t / time_duration) % 1) + self.gaussians.time_duration[0]
        rendered = self.render_fn(self.view, self.gaussians, self.pipeline, self.background)["render"]

        if self.mode in [1, 3, 4]:
            image = rendered
            image = torch.clamp(image, 0, 1)
            image = image * 255
            image = image.to(torch.uint8)
            image = image.permute(1, 2, 0)
            image = image.cpu()  
            gt_image = self.view.image.permute(1, 2, 0).cpu()
            if self.mode in [3]:
                concat_image = np.concatenate((image.numpy(), gt_image.numpy()), axis=1)
                rgb_image = cv2.cvtColor(concat_image, cv2.COLOR_BGR2RGB)
            elif self.mode in [1, 4]:
                rgb_image = cv2.cvtColor(image.numpy(), cv2.COLOR_BGR2RGB)
        elif self.mode == 2 :
            rendered = self.render_fn(self.view, self.gaussians, self.pipeline, self.background)["depth"].detach()[0]

            rendered = rendered.cpu().numpy() 
            normalized_depth = (rendered - np.min(rendered)) / (np.max(rendered) - np.min(rendered))
            rgb_image = cv2.applyColorMap(np.uint8(normalized_depth * 255), cv2.COLORMAP_JET)
    
        return rgb_image
    
    def setView(self, i):
        init_view = self.viewSet[i][1]
        # if R is not torch.tensor
        if not isinstance(init_view.R, torch.Tensor):
            init_view.R = torch.from_numpy(init_view.R).float()
            init_view.T = torch.from_numpy(init_view.T).float()
        
        init_view = init_view.cuda()
        if init_view.image_width != self.width or init_view.image_height != self.height:
            init_view.image_width = self.width
            init_view.focal_x = math.tan(init_view.FoVx / 2.0) / 2.0 * init_view.image_width
            init_view.image_height = self.height
            init_view.focal_y = math.tan(init_view.FoVy / 2.0) / 2.0 * init_view.image_height
            init_view.image = TF.resize(init_view.image, (self.height, self.width))

        self.view = init_view
        print(i, self.view.image_width, self.view.image_height)

    def setViewPos(self, d_camera_pos):
        self.view.T -= d_camera_pos # T
        Rt = torch.zeros((4, 4), device="cuda")
        Rt[:3, :3] = self.view.R.transpose(0,1)
        Rt[:3, 3] = self.view.T
        Rt[3, 3] = 1.0
        self.view.world_view_transform = Rt.transpose(0, 1) # world view transform
        self.view.full_proj_transform = (self.view.world_view_transform.unsqueeze(0).bmm(self.view.projection_matrix.unsqueeze(0))).squeeze(0) # full projection transform
        self.view.camera_center = self.view.world_view_transform.inverse()[3, :3] # camera center

    def setViewRot(self, d_camera_rot):
        self.view.R = (d_camera_rot @ self.view.R.transpose(0,1)).transpose(0,1) # R
        self.view.T = - self.view.R.transpose(0,1) @ self.view.camera_center # T
        Rt = torch.zeros((4, 4), device="cuda")
        Rt[:3, :3] = self.view.R.transpose(0,1)
        Rt[:3, 3] = self.view.T
        Rt[3, 3] = 1.0
        self.view.world_view_transform = Rt.transpose(0, 1) # world view transform
        self.view.full_proj_transform = (self.view.world_view_transform.unsqueeze(0).bmm(self.view.projection_matrix.unsqueeze(0))).squeeze(0) # full projection transform
        
    def setViewFov(self, d_focal_x):
        view = self.view
        focal_x = math.tan(view.FoVx / 2.0) / 2.0 * view.image_width
        focal_y = math.tan(view.FoVy / 2.0) / 2.0 * view.image_height
        ratio = focal_y / focal_x
        focal_x += d_focal_x
        focal_y += d_focal_x * ratio
        view.FoVx = math.atan(focal_x / view.image_width * 2.0) * 2.0
        view.FoVy = math.atan(focal_y / view.image_height * 2.0) * 2.0
        if view.cx > 0:
            view.projection_matrix = getProjectionMatrixCenterShift(view.znear, view.zfar, view.cx, view.cy, view.fl_x, view.fl_y, view.image_width, view.image_height).transpose(0,1).cuda()
        else:
            view.projection_matrix = getProjectionMatrix(znear=view.znear, zfar=view.zfar, fovX=view.FoVx, fovY=view.FoVy).transpose(0,1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0) # full projection transform
        self.view = view
        
    def setViewSize(self, image_width, image_height):
        self.width = image_width
        self.height = image_height
        self.view.image_width = image_width
        self.view.image_height = image_height
        self.view.image = TF.resize(self.view.image, (image_height, image_width))
        

class MainWindow(QMainWindow):
    def __init__(self, *args, mode, fixed):
        super().__init__()
        self.setWindowTitle('3D Browser with PyQt')
        self.image_label = QLabel(self)
        self.setCentralWidget(self.image_label)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.renderScene = RenderScene4DGS(*args, mode, fixed)
        self.mode = mode
        self.fixed = fixed
        if self.mode in [3]:
            self.resize(self.renderScene.view.image_width * 2, self.renderScene.view.image_height)
        else:
            self.resize(self.renderScene.view.image_width, self.renderScene.view.image_height)

        # [you can modify the control sensitivity here]
        self.moving_speed = 0.2 
        self.angel_speed = 0.1 
        self.fov_speed = 0.05 

        self.last_mouse_position = None
        self.mouse_type = None
        self.setMouseTracking(True)

        self.start_time = QDateTime.currentDateTime()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image)
        self.timer.start(66)

    def update_image(self):
        current_time = QDateTime.currentDateTime()
        frame_ratio = 1
        elapsed_time = self.start_time.msecsTo(current_time) / 1000.0 * frame_ratio
        img_data = self.renderScene(elapsed_time)  # the return should be: (height, width, 3), [0, 255], uint8, CPU
        height, width, _ = img_data.shape
        bytes_per_line = 3 * width
        q_img = QImage(img_data.data, width, height, bytes_per_line, QImage.Format_RGB888)
        # q_img = QImage(img_data.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)

    def keyPressEvent(self, event):
        if self.fixed:
            return
        key = event.key()
        d_camera_pose = torch.tensor([0.0, 0.0, 0.0], device="cuda")

        #     ^ y 
        #     |
        #     |
        #     |___________> x
        #     / 
        #    /
        #   /
        #  < -z

        if key == Qt.Key_Q: # +Z
            d_camera_pose[1] -= self.moving_speed
        elif key == Qt.Key_E: # -Z
            d_camera_pose[1] += self.moving_speed
        elif key == Qt.Key_A: # -X
            d_camera_pose[0] -= self.moving_speed
        elif key == Qt.Key_D: # +X
            d_camera_pose[0] += self.moving_speed
        elif key == Qt.Key_W: # -Y
            d_camera_pose[2] += self.moving_speed
        elif key == Qt.Key_S: # +Y
            d_camera_pose[2] -= self.moving_speed
        print("Pose: ", d_camera_pose.tolist())
        self.renderScene.setViewPos(d_camera_pose)

    def mousePressEvent(self, event):
        if self.fixed:
            return
        self.mouse_type = event.button()
        self.last_mouse_position = event.pos()

    def mouseReleaseEvent(self, event):
        if self.fixed:
            return
        self.mouse_type = None
        self.last_mouse_position = None

    def mouseMoveEvent(self, event):
        if self.fixed:
            return
        if self.mouse_type == Qt.LeftButton:
            delta = event.pos() - self.last_mouse_position   
            x_rad = torch.deg2rad(torch.tensor(delta.x() * self.angel_speed, device="cuda"))
            y_rad = torch.deg2rad(torch.tensor(-delta.y() * self.angel_speed, device="cuda"))
            print("Angel: ", x_rad.item(), y_rad.item())
            Rx = torch.tensor([
                [1, 0, 0],
                [0, torch.cos(y_rad), -torch.sin(y_rad)],
                [0, torch.sin(y_rad), torch.cos(y_rad)]
            ], device="cuda")
            Ry = torch.tensor([
                [torch.cos(x_rad), 0, torch.sin(x_rad)],
                [0, 1, 0],
                [-torch.sin(x_rad), 0, torch.cos(x_rad)]
            ], device="cuda")
            R = Ry @ Rx
            self.renderScene.setViewRot(R)
            self.last_mouse_position = event.pos()
        elif self.mouse_type == Qt.RightButton:
            delta = event.pos() - self.last_mouse_position   
            delta = (delta.y() + delta.x()) * self.angel_speed
            z_rad = torch.deg2rad(torch.tensor(delta, device="cuda"))
            print("Angel: ", z_rad.item())
            Rz = torch.tensor([
                [torch.cos(z_rad), -torch.sin(z_rad), 0],
                [torch.sin(z_rad), torch.cos(z_rad), 0],
                [0, 0, 1]
            ], device="cuda")
            R = Rz
            self.renderScene.setViewRot(R)
            self.last_mouse_position = event.pos()

    def wheelEvent(self, event):
        if self.fixed:
            return
        delta = event.angleDelta().y() * self.fov_speed
        d_focal = -delta
        print("Fov: ", d_focal)
        self.renderScene.setViewFov(d_focal)
    
    def resizeEvent(self, event):
        image_width = self.image_label.width()
        image_height = self.image_label.height()
        if self.mode in [3]:
            self.renderScene.setViewSize(image_width // 2, image_height)
        else:
            self.renderScene.setViewSize(image_width, image_height)
        super().resizeEvent(event)
            