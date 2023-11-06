from mmhuman3d.utils.transforms import rotmat_to_aa
from pytorch3d.renderer.cameras import look_at_view_transform
import joblib
import torch
from avatar3d.cameras import build_cameras
from avatar3d.cameras.convert_convention import convert_K_3x3_to_4x4
from pytorch3d.structures import Meshes
import numpy as np
from torch.utils.data import Dataset


class DistortionMapDataset(Dataset):

    def __init__(self, data_path, sample_num=None, test_mode=False) -> None:
        with open(data_path, 'rb') as f:
            self.db = joblib.load(f)
        self.sample_num = sample_num
        self.test_mode = test_mode

    def __len__(self):
        return len(self.db['img_name']) if self.sample_num is None else int(
            self.sample_num)

    def __getitem__(self, index):
        poses = torch.Tensor(self.db['pose_refined'][index, :66])
        betas = torch.Tensor(self.db['shape'][index])
        left_hand_pose = torch.Tensor(self.db['hand_pose'][index, 3:48])
        right_hand_pose = torch.Tensor(self.db['hand_pose'][index, 51:96])
        body_pose = poses[3:]
        flag = np.random.rand()

        if flag <= 0.5:
            # random
            s = torch.Tensor(np.random.uniform(0.5, 1.0, size=(1, 1)), )
            tx = torch.Tensor(
                np.random.uniform(0.5 - 1 / s, 1 / s - 0.5, size=(1, 1)), )
            # sty = torch.Tensor(
            #     np.random.uniform(0.1, 0.8, size=(num_sample_per_obj, 1)), )
            ty = torch.Tensor(
                np.random.uniform(0.65 - 1 / s, 1 / s - 0.5, size=(1, 1)))
            # ty = sty / s
            cam = torch.cat([s, tx, ty], -1)
            global_orient = orbit_orient(1)[0]
            K = random_K(batch_size=1, uniform=True)
        elif flag <= 0.7:
            # halfbody rotate
            s = torch.Tensor(np.random.uniform(0.8, 1.6, size=(1, 1)))
            ty = torch.Tensor(
                np.random.uniform(0.9 - 1 / s, 1 / s, size=(1, 1)))
            tx = torch.Tensor(
                np.random.uniform(0.5 - 1 / s, 1 / s - 0.5, size=(1, 1)), )
            cam = torch.cat([s, tx, ty], -1)
            global_orient = orbit_orient(1)[0]
            K = random_K(batch_size=1, uniform=True)
        elif flag <= 1:
            # halfbody
            s = torch.Tensor(np.random.uniform(1.6, 2, size=(1, 1)))
            ty = torch.Tensor(
                np.random.uniform(0.9 - 1 / s, 1 / s, size=(1, 1)))
            tx = torch.Tensor(
                np.random.uniform(0.5 - 1 / s, 1 / s - 0.5, size=(1, 1)), )
            cam = torch.cat([s, tx, ty], -1)
            global_orient = poses[:3]
            K = random_K(batch_size=1, uniform=False)

        cam = cam.view(3)
        f = K[0, 0, 0].view(1)

        transl_z = f / cam[0:1]
        transl = torch.cat([cam[1:3], transl_z], -1)
        data = dict(
            transl=transl,
            cam=cam,
            K=K[0],
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
        )
        return data


def random_K(batch_size=1, uniform=True):
    if uniform:
        fov = np.random.uniform(low=10, high=120, size=batch_size)
    else:
        fov = np.random.normal(loc=80, scale=40, size=batch_size)
    fov = np.clip(fov, 5, 120)
    f = np.tan(np.pi / 2 - np.deg2rad(fov / 2))
    K = np.eye(3, 3)[None].repeat(batch_size, 0)
    K[:, 0, 0] = f
    K[:, 1, 1] = f
    K = convert_K_3x3_to_4x4(torch.Tensor(K))
    return K


def random_camera(batch_size=1,
                  resolution=(320, 320),
                  device=None,
                  uniform=True):
    if uniform:
        fov = np.random.uniform(low=10, high=120, size=batch_size)
    else:
        fov = np.random.normal(loc=80, scale=40, size=batch_size)
    fov = np.clip(fov, 5, 120)
    f = np.tan(np.pi / 2 - np.deg2rad(fov / 2))
    K = np.eye(3, 3)[None].repeat(batch_size, 0)
    K[:, 0, 0] = f
    K[:, 1, 1] = f
    K = convert_K_3x3_to_4x4(torch.Tensor(K))
    cameras = build_cameras(
        dict(type='perspective',
             K=K,
             in_ndc=True,
             resolution=resolution,
             convention='opencv'))
    return cameras.to(device)


def orbit_orient(batch_size=1):
    elev = torch.Tensor(np.random.uniform(-20, 40, size=(batch_size, 1)))
    azim = torch.Tensor(np.random.normal(loc=0, scale=90,
                                         size=(batch_size, 1)))
    dist = torch.Tensor(np.random.uniform(1., 10, size=(batch_size, 1)))
    R, _ = look_at_view_transform(elev=elev,
                                  azim=azim,
                                  dist=dist,
                                  up=((0, -1, 0), ))
    R = R.permute(0, 2, 1)
    global_orient = rotmat_to_aa(R)
    return global_orient


def render(global_orient,
           body_pose,
           betas,
           cam,
           cameras,
           renderer,
           device,
           body_model,
           left_hand_pose=None,
           right_hand_pose=None):

    cam = cam.to(device)
    betas = betas.to(device)
    body_pose = body_pose.to(device)
    global_orient = global_orient.to(device)

    left_hand_pose = left_hand_pose.to(
        device) if left_hand_pose is not None else None
    right_hand_pose = right_hand_pose.to(
        device) if right_hand_pose is not None else None
    batch_size = body_pose.shape[0]

    f = cameras.K[:, 0, 0].unsqueeze(-1)

    transl_z = f / cam[..., 0:1]
    transl = torch.cat([cam[..., 1:3], transl_z], -1)

    body_model_output = body_model(global_orient=global_orient,
                                   body_pose=body_pose,
                                   left_hand_pose=left_hand_pose,
                                   right_hand_pose=right_hand_pose,
                                   betas=betas,
                                   transl=transl)
    verts = body_model_output['vertices']

    faces = body_model.faces_tensor[None].repeat_interleave(batch_size, 0)
    meshes = Meshes(verts, faces)
    depth_map = renderer(meshes, cameras)

    mask = (depth_map > 0).float()
    depth_map = depth_map * mask + 1 * (1 - mask)

    distortion_map = transl_z.view(batch_size, 1, 1, 1) / depth_map[..., 0:1]
    distortion_map = distortion_map * mask
    # meshes.textures =
    return distortion_map, transl
