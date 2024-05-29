## How to use
It's for waymo open dataset and real-time rendering (low FPS).

## Install
CLI
```shell
conda create -n $env_name python=$version
# install torch with correct version
pip install -r requirements.txt
pip install ./diff-gaussian-rasterization
pip install ./pointops2
pip install ./simple-knn
```

## Setup
1. Download individual scene in waymo open dataset scene flow labels

2. Use EmerNeRF to preprocess the individual scene, the preprocessed scene data folder should be named as xxx (e.g. 000, 001, 002)

3. Modify the `./config/waymo/xxx.yaml`, especially `source_path` (input) and `model_path` (output)


## Run
**Train** (no implementation for load checkpoint and continue train):
```shell
python train.py --config configs/waymo/xxx.yaml
```

**Render**:
```shell
python eval.py --config configs/waymo/xxx.yaml --pth output_path/xxx/chkpntxxx.pth --mode 0 --viewDir 0
```
###### Mode:
1. render point cloud with random color in real time
2. render depth map in real time
3. render training camera trace in real time
4. render free camera in real time 
5. save training camera trace as video
6. save depth map as video
7. evaluate metrics (PSNR, SSIM, LPIPS) 

###### Interactivation:
Mode 1, 4 support interacitve real time rendering  
W S A D : move forward/backward/left/right  
Q E: move upwards/downwards  
Hold left or right mouse: rotation  
Roll: change focal  
Resize window: the render resolution also resize  

###### How to render novel view video:
1. Use interacitve window and select a new view in your local computer.
2. Close the window, and a `view.obj` file would be saved in working directory.
3. Move the file to server working directory
4. render with mode 5/6 



