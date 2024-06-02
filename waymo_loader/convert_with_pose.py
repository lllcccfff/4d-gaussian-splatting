# Import necessary library
import numpy as np
from scipy.spatial.transform import Rotation
import os
import sys
import sqlite3

from raw_loader import load_ego_pose, load_image, load_intrinsics, load_extrinsics

# Define the paths for the output files
cam_path = 'cameras.txt'
img_path = 'images.txt'
pt_path = 'points3D.txt'
opencv2waymo = np.array([[0, 0, 1, 0], 
                        [-1, 0, 0, 0], 
                        [0, -1, 0, 0], 
                        [0, 0, 0, 1]])

def write_to_colmap(base_dir):
    ego_pose = load_ego_pose(os.path.join(base_dir, "ego_pose"))
    images, img_pth = load_image(os.path.join(base_dir, "images"))
    intrinsics = load_intrinsics(os.path.join(base_dir, "intrinsics"))
    extrinsics = load_extrinsics(os.path.join(base_dir, "extrinsics"))

    colmap_path = os.path.join(base_dir, 'colmap')
    if not os.path.exists(colmap_path):
        os.makedirs(colmap_path)
    with open(os.path.join(colmap_path, cam_path), 'w') as cam_txt, open(os.path.join(colmap_path, img_path), 'w') as image_txt, open(os.path.join(colmap_path, pt_path), 'w') as pt_txt:
        frame_num = len(images)
        camera_num = len(images[0])
        image_num = frame_num * camera_num

        cam_txt.write('#By liminghao\n#CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n# Number of cameras: {} \n'.format(image_num))
        image_txt.write('#By liminghao\n#IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n# Number of images: {} \n'.format(image_num))
        
        for i in range(frame_num):
            for j in range(camera_num):
                cam_idx = 1 + i * camera_num + j
                h, w, c = images[i][j].shape
                fx, fy, cx, cy, k1, k2, p1, p2, k3 = intrinsics[j]

                cam_txt.write('{} PINHOLE {} {} {} {} {} {}\n'.format(cam_idx, w, h, fx, fy, cx, cy))
                image_txt.write('{} '.format(cam_idx))

                c2w = ego_pose[i] @ extrinsics[j] @ opencv2waymo
                w2c = np.linalg.inv(c2w)
                R = w2c[:3, :3]
                T = w2c[:3, 3]
                rot = Rotation.from_matrix(R)
                Q = rot.as_quat()

                image_txt.write('{} '.format(Q[3]))
                image_txt.write('{} '.format(Q[0]))
                image_txt.write('{} '.format(Q[1]))
                image_txt.write('{} '.format(Q[2]))

                for ti in T:
                    image_txt.write('{} '.format(ti))
                
                # Write camera ID and image filename to the image file
                image_txt.write('{} {}\n\n'.format(cam_idx, img_pth[cam_idx - 1]))

IS_PYTHON3 = sys.version_info[0] >= 3
def array_to_blob(array):
    if IS_PYTHON3: return array.tostring()
    else: return np.getbuffer(array)
def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3: return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else: return np.frombuffer(blob, dtype=dtype).reshape(*shape)
class COLMAPDatabase(sqlite3.Connection):
    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)
        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=True WHERE camera_id=?",
            (model, width, height, array_to_blob(params),camera_id))
        return cursor.lastrowid
def camTodatabase(base_dir):
    cameras_txt_path = os.path.join(base_dir, 'colmap', 'cameras.txt')
    db_path = os.path.join(base_dir, 'colmap', 'database.db')
    camModelDict = {'SIMPLE_PINHOLE': 0,
                    'PINHOLE': 1,
                    'SIMPLE_RADIAL': 2,
                    'RADIAL': 3,
                    'OPENCV': 4,
                    'FULL_OPENCV': 5,
                    'SIMPLE_RADIAL_FISHEYE': 6,
                    'RADIAL_FISHEYE': 7,
                    'OPENCV_FISHEYE': 8,
                    'FOV': 9,
                    'THIN_PRISM_FISHEYE': 10}
    db = COLMAPDatabase.connect(db_path)

    idList=list()
    modelList=list()
    widthList=list()
    heightList=list()
    paramsList=list()
    # Update real cameras from .txt
    with open(cameras_txt_path, "r") as cam:
        lines = cam.readlines()
        for i in range(0,len(lines),1):
            if lines[i][0]!='#':
                strLists = lines[i].split()
                cameraId=int(strLists[0])
                cameraModel=camModelDict[strLists[1]] #SelectCameraModel
                width=int(strLists[2])
                height=int(strLists[3])
                paramstr=np.array(strLists[4:12])
                params = paramstr.astype(np.float64)
                idList.append(cameraId)
                modelList.append(cameraModel)
                widthList.append(width)
                heightList.append(height)
                paramsList.append(params)
                camera_id = db.update_camera(cameraModel, width, height, params, cameraId)

    db.commit()
    rows = db.execute("SELECT * FROM cameras")
    for i in range(0,len(idList),1):
        camera_id, model, width, height, params, prior = next(rows)
        params = blob_to_array(params, np.float64)
        assert camera_id == idList[i]
        assert model == modelList[i] and width == widthList[i] and height == heightList[i]
        assert np.allclose(params, paramsList[i])
    db.close()

# --- --- ---
def creat_SfM_cloud(base_dir): # 'data/waymo/023'
    write_to_colmap(base_dir)

    # load given camera poses
    colmap_cmd = "colmap"
    colmap_path = os.path.join(base_dir, 'colmap')
    db_path = os.path.join(base_dir, "colmap", "database.db")
    img_path = os.path.join(base_dir, "images")

    # Feature extraction
    feature_extracton_cmd = colmap_cmd + " feature_extractor" + \
        " --database_path " + db_path + \
        " --image_path " + img_path + \
        " --ImageReader.camera_model PINHOLE"
    exit_code = os.system(feature_extracton_cmd)
    if exit_code != 0:
        print(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)
    print("1. Feature extraction successed.")
    
    # load to database
    camTodatabase(base_dir)

    # Feature matching
    feat_matching_cmd = colmap_cmd + " exhaustive_matcher" + \
       " --database_path " + db_path
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        print(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)
    print("2. Feature matching successed.")

    # Triangulator
    triangulator_cmd = colmap_cmd + " point_triangulator" + \
       " --database_path " + db_path + \
        " --image_path " + img_path + \
        " --input_path " + colmap_path + \
        " --output_path " + colmap_path
    exit_code = os.system(triangulator_cmd)
    if exit_code != 0:
        print(f"Triangulator failed with code {exit_code}. Exiting.")
        exit(exit_code)
    print("3. Triangulator successed.")

    # Convert bin to txt
    command = colmap_cmd + " model_converter" + \
            " --input_path " + colmap_path + \
            " --output_path " + colmap_path + \
            " --output_type TXT"
    os.system(command)
    print("4. Txt data created.")

