import os, logging
import shutil

def convert(src_pth):
    if os.path.exists(os.path.join(src_pth, "sparse", "0")):
        print("All needed data already exists.")
        return

    distorted_pth = os.path.join(src_pth, "distorted", "sparse")
    db_path = os.path.join(src_pth, "distorted", "database.db")
    input_img_path = os.path.join(src_pth, "input")

    colmap_cmd = "colmap"
    os.makedirs(distorted_pth, exist_ok=True)
    # Feature extraction
    feature_extracton_cmd = colmap_cmd + " feature_extractor" + \
        " --database_path " + db_path + \
        " --image_path " + input_img_path +  \
        " --ImageReader.single_camera 1" + \
        " --ImageReader.camera_model OPENCV"
    exit_code = os.system(feature_extracton_cmd)
    if exit_code != 0:
        print(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)
    print("1. Feature extraction successed.")

    # Feature matching
    feat_matching_cmd = colmap_cmd + " exhaustive_matcher" + \
       " --database_path " + db_path
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        print(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)
    print("2. Feature matching successed.")

    # Bundle adjustment
    mapper_cmd = colmap_cmd + " mapper" + \
        " --database_path " + db_path + \
        " --image_path "  + input_img_path + \
        " --output_path "  + distorted_pth + \
        " --Mapper.ba_global_function_tolerance=0.000001"
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        print(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)
    print("3. Bundle adjustment successed.")

    ### Image undistortion
    img_undist_cmd = colmap_cmd + " image_undistorter" + \
        " --image_path " + input_img_path + \
        " --input_path " + os.path.join(distorted_pth, "0") + \
        " --output_path " + src_pth + \
        " --output_type COLMAP"
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        print(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)
    print("4. Image undistortion successed.")

    # move to expected location
    files = os.listdir(src_pth + "/sparse")
    final_pth = os.path.join(src_pth, "sparse", "0")
    os.makedirs(final_pth, exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(src_pth, "sparse", file)
        destination_file = os.path.join(final_pth, file)
        shutil.move(source_file, destination_file)

    # convert .bin to .txt
    # files = os.listdir(final_pth)
    # for file in files:
    #     if file.endswith('.bin'):
    #         bin_path = os.path.join(final_pth, file)
    #         txt_path = os.path.join(final_pth, file.replace('.bin', '.txt'))

    command = colmap_cmd + " model_converter" + \
        " --input_path " + final_pth + \
        " --output_path " + final_pth + \
        " --output_type TXT"
    os.system(command)
    print("5. .txt data created.")

if __name__ == "__main__":
    source_path = "data\waymo\segment-104481_colmap"
    convert(source_path)
