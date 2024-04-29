## How to use
It's for waymo open dataset and real-time rendering (low FPS).

## Install
CLI
```shell
conda create -n $env_name python=$version
# install torch with correct version
pip install -r requirements.txt
pip install diff-gaussian-rasterization
pip install pointops2
pip install simple-knn
```

## Setup
1. Download individual scene in waymo open dataset scene flow labels

2. Use EmerNeRF to preprocess the individual scene, the preprocessed scene data folder should be named as xxx (e.g. 000, 001, 002)

3. Modify the `./config/waymo/xxx.yaml`, especially `source_path` (input) and `model_path` (output)

4. Move your data

## Run
**Train** (no implementation for load checkpoint and continue train):
```shell
python train.py --config configs/waymo/xxx.yaml
```

**Render**:
```shell
python render.py --config configs/waymo/xxx.yaml --pth output_path/xxx/chkpnt_best.pth
```

> If necessary, i would develop a convenient way to switch different mode before render. (point cloud/depth map/fix camera and trace/free camera)

