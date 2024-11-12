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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import torch
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from scene.ellipse_utils import generate_ellipse_path_from_camera_infos
from utils.camera_utils import CameraInfo
import pickle
from tqdm import tqdm

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    render_cameras: list
    nerf_normalization: dict
    ply_path: str

def load_intrinsics_pickle(path):
    with open(path, 'rb') as f:
        intrinsics = pickle.load(f)['intrinsics']
    return intrinsics

def load_poses_pickle(path): #read poses in pytorch3d format
    with open(path, 'rb') as f:
        camera_poses = pickle.load(f)['poses']
    poses = {}

    for key in tqdm(camera_poses.keys(), desc='Reading cameras'):
            poses[key] = torch.from_numpy(camera_poses[key])
        
    return poses

def opencv_from_cameras_projection(camera_pose):
    R_pytorch3d = camera_pose[:3, :3].clone()
    T_pytorch3d = camera_pose[:3, 3].clone()

    T_pytorch3d[:2] *= -1
    R_pytorch3d[:, :2] *= -1
    tvec = T_pytorch3d

    R = R_pytorch3d.permute(1, 0)
    rt = torch.cat([R, tvec.unsqueeze(1)], dim=1)
    rt = torch.cat([rt, torch.tensor([[0, 0, 0, 1]], device=rt.device)], dim=0)
    
    return rt.cpu().numpy()

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readPytorchSceneInfo(path, eval, all_args, llffhold=8):
    pickle_path_train = os.path.join(path, "images/train/camera_train.pickle")
    pickle_path_test = os.path.join(path, "images/test/camera_test.pickle")

    cam_intrinsics = load_intrinsics_pickle(pickle_path_train) #read intrinsics from pytorch pickle file
    cam_extrinsics_train = load_poses_pickle(pickle_path_train) #read poses from pytorch pickle file and return in opencv/colmap format
    cam_extrinsics_test = load_poses_pickle(pickle_path_test) 

    reading_dir = "images" 
    cam_infos_unsorted_train = readPytorchCameras(cam_extrinsics=cam_extrinsics_train, cam_intrinsics=cam_intrinsics, 
                                           images_folder=os.path.join(path, reading_dir), split='train')
    cam_infos_train = sorted(cam_infos_unsorted_train.copy(), key = lambda x : x.image_name)

    cam_infos_unsorted_test = readPytorchCameras(cam_extrinsics=cam_extrinsics_test, cam_intrinsics=cam_intrinsics, 
                                           images_folder=os.path.join(path, reading_dir), split='test')
    cam_infos_test = sorted(cam_infos_unsorted_test.copy(), key = lambda x : x.image_name)

    #img_to_keep = ['000.png', '015.png', '030.png', '045.png']
    #img_to_keep = ['000.jpg', '041.jpg', '082.jpg', '125.jpg']
    #img_to_keep = ['000.png', '015.png', '030.png', '045.png','060.png', '075.png', '090.png', '105.png']
    #train_cam_infos = [cam_info for cam_info in cam_infos_train if cam_info.image_name in img_to_keep]
    train_cam_infos = cam_infos_train
    test_cam_infos = cam_infos_test

    render_cam_infos = generate_ellipse_path_from_camera_infos(cam_infos_train)

    nerf_normalization = getNerfppNorm(train_cam_infos)
    print(nerf_normalization)
    ply_path = all_args.ply_path
    if not os.path.exists(ply_path):
        ply_path = os.path.join(path, "sparse_colmap/0/points3D.ply")
    bin_path = os.path.join(path, "sparse_colmap/0/points3D.bin")
    txt_path = os.path.join(path, "sparse_colmap/0/points3D.txt")

    
    if all_args.random_ply:
        #print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        #try:
        #    xyz, rgb, _ = read_points3D_binary(bin_path)
        #except:
        #    xyz, rgb, _ = read_points3D_text(txt_path)
        #storePly(ply_path, xyz, rgb)
            # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        
        ply_path = os.path.join(path, "sparse_colmap/0/points3D_random.ply")
        storePly(ply_path, xyz, SH2RGB(shs) * 255)

        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None

    print('Reading ply file from: {0}'.format(ply_path))
    pcd = fetchPly(ply_path)

    #train_cam_infos = random.sample(train_cam_infos, k=int(len(train_cam_infos)*all_args.image_perc))
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           render_cameras=render_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    print('Num of train cameras: {0}'.format(len(train_cam_infos))) 
    print('Num of test cameras: {0}'.format(len(test_cam_infos))) 

    return scene_info


def readPytorchCameras(cam_extrinsics, cam_intrinsics, images_folder, split):
    cam_infos = []

    images_files = sorted(os.listdir(images_folder + '/' + split + '/images'))

    width = int(cam_intrinsics[0,2] * 2)
    height = int(cam_intrinsics[1,2] * 2)

    focal_length_x = cam_intrinsics[0,0]
    FovY = focal2fov(focal_length_x, height)
    FovX = focal2fov(focal_length_x, width)

    depth_path = images_folder + '/' + split + '/depths'
    weight_path = images_folder + '/' + split + '/weights'

    for i, n in enumerate(sorted(cam_extrinsics.keys())):
        curr_pose = cam_extrinsics[n]

        image_path = os.path.join(images_folder + '/' + split + '/images', n)
        image_name = os.path.basename(image_path)#.split(".")[0]
        image = Image.open(image_path)
        
        curr_pose = opencv_from_cameras_projection(curr_pose).astype(np.float64)

        R = curr_pose[:3, :3]
        R = np.transpose(R)
        T = curr_pose[:3, 3]

        mask_path = os.path.join(images_folder + '/' + split + '/masks', images_files[i].replace('png','npy'))
        if os.path.exists(mask_path):
            mask = np.load(mask_path).astype(np.uint8)
        else:
            mask = None 

        if os.path.exists(depth_path):
            #depth_map = cv2.imread(os.path.join(depth_path, images_files[i].replace('png','npy')), cv2.IMREAD_ANYDEPTH) / 1000
            depth_map = np.load(os.path.join(depth_path, images_files[i].replace('png','npy')))#.astype(np.float64)
        else:
            depth_map = None

        if os.path.exists(weight_path):
            #depth_map = cv2.imread(os.path.join(depth_path, images_files[i].replace('png','npy')), cv2.IMREAD_ANYDEPTH) / 1000
            weight_map = np.load(os.path.join(weight_path, images_files[i].replace('png','npy')))#.astype(np.float64)
        else:
            weight_map = None

        cam_info = CameraInfo(uid=n, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=width, height=height, 
                                image_mask=mask, mask_path=mask_path, depth_map=depth_map, weight_map=weight_map)

        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None


    #ply_path = os.path.join(path, "points3d.ply")
    #if not os.path.exists(ply_path):
    # Since this data set has no colmap data, we start with random points
    #num_pts = 100_000
    #print(f"Generating random point cloud ({num_pts})...")
    #    
    #    # We create random points inside the bounds of the synthetic Blender scenes
    #xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
    #shs = np.random.random((num_pts, 3)) / 255.0
    #pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    #storePly(ply_path, xyz, SH2RGB(shs) * 255)
    #try:
    #    pcd = fetchPly(ply_path)
    #except:
    #    pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], mask_path=''))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)

    img_to_keep = ['rgb_000', 'rgb_050', 'rgb_100', 'rgb_150']
    train_cam_infos = [cam_info for cam_info in train_cam_infos if cam_info.image_name in img_to_keep]
    
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    render_cam_infos = generate_ellipse_path_from_camera_infos(train_cam_infos)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                            render_cameras=render_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Pytorch" : readPytorchSceneInfo
}