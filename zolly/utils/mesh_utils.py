import torch
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.structures import (Meshes, join_meshes_as_scene)
from mmhuman3d.utils.mesh_utils import (join_batch_meshes_as_scene,
                                        load_plys_as_meshes,
                                        load_objs_as_meshes,
                                        save_meshes_as_objs,
                                        save_meshes_as_plys)

from pytorch3d.utils.ico_sphere import ico_sphere
from pytorch3d.renderer.mesh import TexturesVertex
from zolly.utils.keypoint_utils import search_limbs

from mmhuman3d.core.conventions.segmentation import body_segmentation
from mmhuman3d.utils.demo_utils import get_different_colors
from .cylindar import cylindar


def get_pointcloud_mesh(verts_padded, level=0, radius=0.01, colors=None):
    device = verts_padded.device
    B, N, _ = verts_padded.shape  # B, N, 3
    sphere = ico_sphere(level).to(device)
    n = sphere.verts_padded().shape[1]
    f = sphere.faces_padded().shape[1]
    verts = radius * sphere.verts_padded()[:, None].repeat(B, N, 1,
                                                           1)  #B, N, n, 3
    faces = sphere.faces_padded()[:, None].repeat(B, N, 1, 1)  #B, N, f, 3
    faces_offsets = torch.arange(0, N)[None, :, None, None].repeat(
        B, 1, 1, 1) * n  # B, N, 1, 1
    verts = verts + verts_padded[:, :, None]
    faces = faces + faces_offsets.to(device)
    textures = None
    if colors is not None:
        colors = colors.to(device)[:, :, None].repeat_interleave(n,
                                                                 2)  #B, N,n, 3

        colors = colors.view(B, N * n, 3)
        textures = TexturesVertex(colors)
    verts = verts.view(B, N * n, 3)
    faces = faces.view(B, N * f, 3)
    meshes = Meshes(verts=verts, faces=faces, textures=textures)
    return meshes


def get_cylinder(start, end, radius):

    def norm_vec(vec):
        return vec / torch.sqrt((vec * vec).sum())

    device = start.device
    mesh = Meshes(verts=torch.Tensor(cylindar['verts'])[None],
                  faces=torch.Tensor(
                      cylindar['faces'])[None].long()).to(device)
    verts = mesh.verts_padded()
    length = torch.sqrt(((end - start) * (end - start)).sum())
    verts[..., :2] *= radius
    verts[..., 2] = verts[..., 2] / 1.1336 / 2 * length
    center = start / 2 + end / 2

    z_vec = end - start
    z_vec = z_vec.view(-1, 3)
    z_vec = norm_vec(z_vec)
    y_vec = torch.Tensor([0, 1, 0]).view(-1, 3).to(device)

    y_vec = y_vec - torch.bmm(y_vec.view(-1, 1, 3), z_vec.view(-1, 3, 1)).view(
        -1, 1) * z_vec
    y_vec = norm_vec(y_vec)
    x_vec = torch.cross(y_vec, z_vec)
    R = torch.cat(
        [x_vec.view(-1, 3, 1),
         y_vec.view(-1, 3, 1),
         z_vec.view(-1, 3, 1)], 1).view(-1, 3, 3)

    R = R.permute(0, 2, 1)
    n_verts = verts.shape[1]
    verts = torch.bmm(R.repeat_interleave(n_verts, 0), verts.view(-1, 3, 1))
    verts = verts.view(1, -1, 3)
    verts = verts + center.view(1, 1, 3)
    mesh = mesh.update_padded(verts)
    mesh.textures = TexturesVertex(
        torch.ones_like(verts) *
        torch.Tensor([0.3, 0.3, 0.8]).view(1, 1, 3).to(device))
    return mesh


def get_joints_mesh(joints, level=0, radius=0.01, colors=None):
    colors = torch.randint(0, 255, joints.shape).float()
    colors = colors / 255.
    pc_mesh = get_pointcloud_mesh(joints,
                                  level=level,
                                  radius=radius,
                                  colors=colors)
    limbs = search_limbs('smpl')[0]['body']
    limbs_list = []
    for limb in limbs:
        start = joints[0, limb[0]]
        end = joints[0, limb[1]]
        limbs_list.append(get_cylinder(start, end, 0.05))
    pc_mesh = join_meshes_as_scene(limbs_list + [pc_mesh])
    return pc_mesh


def get_smpl_mesh(verts, faces, palette='white'):
    body_segger = body_segmentation('smpl')
    device = verts.device
    if palette == 'white':
        colors = torch.ones_like(verts)
    elif palette == 'part':
        colors = torch.zeros_like(verts)
        color_part = get_different_colors(len(body_segger), int_dtype=False)
        for part_idx, k in enumerate(body_segger.keys()):
            j = body_segger[k]
            colors[:, j] = torch.FloatTensor(color_part[part_idx]).to(device)
    mesh = Meshes(verts, faces, colors=TexturesVertex(colors))
    return mesh


def get_smpl_pc_mesh(verts, level=0, radius=0.01, palette='white'):
    body_segger = body_segmentation('smpl')
    device = verts.device
    if palette == 'white':
        colors = torch.ones_like(verts)
    elif palette == 'part':
        colors = torch.zeros_like(verts)
        color_part = get_different_colors(len(body_segger), int_dtype=False)
        for part_idx, k in enumerate(body_segger.keys()):
            j = body_segger[k]
            colors[:, j] = torch.FloatTensor(color_part[part_idx]).to(device)
    pc_mesh = get_pointcloud_mesh(verts,
                                  level=level,
                                  radius=radius,
                                  colors=colors)
    return pc_mesh
