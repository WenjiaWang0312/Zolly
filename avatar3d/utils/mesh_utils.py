import warnings
from typing import List, Optional, Union
from mmhuman3d.utils.path_utils import prepare_output_path
from packaging import version
import cv2
import numpy as np
# import open3d
from pytorch3d.io.obj_io import save_obj, load_objs_as_meshes
import torch
from pytorch3d.renderer.mesh.textures import TexturesUV, TexturesVertex
from pytorch3d.structures import (Meshes, Pointclouds, join_meshes_as_scene,
                                  list_to_padded)
from pytorch3d.transforms import Rotate
from mmhuman3d.utils.mesh_utils import join_batch_meshes_as_scene, load_plys_as_meshes, load_objs_as_meshes, save_meshes_as_objs, save_meshes_as_plys
from tqdm import trange

from pytorch3d.utils.ico_sphere import ico_sphere
from pytorch3d.renderer.mesh import TexturesVertex
from avatar3d.utils.keypoint_utils import search_limbs

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


# o3d = open3d
# # vec3d = o3d.utility.Vector3dVector
# # vec2d = o3d.utility.Vector2dVector
# # vec3i = o3d.utility.Vector3iVector
# # PointCloud_o3d = o3d.geometry.PointCloud
# # TriangleMesh = o3d.geometry.TriangleMesh

# new_version = version.parse(o3d.__version__) > version.parse('0.9.0')

# def get_oriented_bbox(
#     points: Optional[Union[torch.Tensor, np.ndarray]] = None,
#     meshes: Optional[Meshes] = None,
# ) -> torch.Tensor:
#     """
#     Get oriented bounding box from a batch of meshes or points.

#     Args:
#         points (Optional[Union[torch.Tensor, np.ndarray]], optional):
#             Batch of points, shape should be (batch, N, 3). N must > 8.
#             Defaults to None.
#         meshes (Optional[Meshes], optional):
#             Batch of meshes. Defaults to None.

#     Returns:
#         torch.Tensor: shape would be (batch, 8, 3)
#     """
#     assert points is not None or meshes is not None,\
#         'Please pass correct input.'
#     if meshes is not None:
#         for mesh in meshes:
#             points.append(mesh.verts_padded())
#     if isinstance(points, torch.Tensor):
#         if points.ndim == 2:
#             points = points[None]
#         points = points.detach().cpu().numpy()
#     elif isinstance(points, np.ndarray):
#         if points.ndim == 2:
#             points = points[None]
#     bboxes = []
#     for points_per_batch in points:
#         points_per_batch = vec3d(points_per_batch)
#         bbox = np.asarray(
#             o3d.geometry.OrientedBoundingBox.create_from_points(
#                 points=points_per_batch).get_box_points())
#         bboxes.append(torch.Tensor(bbox))
#     bboxes = torch.cat(bboxes).view(-1, 8, 3)
#     return bboxes


def axis_align_obb_rotation(
    points: Union[torch.FloatTensor,
                  np.ndarray], bbox: Union[torch.FloatTensor, np.ndarray]
) -> Union[torch.FloatTensor, np.ndarray]:
    """[summary]

    Args:
        points (Union[torch.FloatTensor, np.ndarray]): [description]
        bbox (Union[torch.FloatTensor, np.ndarray]): [description]

    Returns:
        Union[torch.FloatTensor, np.ndarray]: [description]
    """
    assert type(points) is type(
        bbox), 'Points and bbox should be the same type'
    device = torch.device('cpu')
    if isinstance(bbox, np.ndarray):
        bbox = torch.Tensor(bbox)
        points = torch.Tensor(points)
        data_type = 'numpy'
    else:
        data_type = 'tensor'
        device = points.device

    def norm(vec):
        vec = vec.view(-1, 3)
        # shape should be (n, 3), return the same shape normed vec
        return vec / torch.sqrt(vec[:, 0]**2 + vec[:, 1]**2 + vec[:, 2]**2)

    cross = np.corss if isinstance(bbox, np.ndarray) else torch.cross
    original_shape = points.shape
    points = points.reshape(-1, 3)
    bbox = bbox.reshape(8, 3)
    diff = bbox[0:1] - bbox
    length = diff[:, 0]**2 + diff[:, 1]**2 + diff[:, 2]**2
    max_bound = length.max() + 1
    length[0] = max_bound
    index1 = int(torch.argmin(length))
    length[index1] = max_bound
    index2 = int(torch.argmin(length))
    axis1 = norm(bbox[index1] - bbox[0])
    axis2 = norm(bbox[index2] - bbox[0])
    axis3 = cross(axis1, axis2)
    rotmat = torch.cat([axis1, axis2, axis3])
    arg = torch.argmax(abs(rotmat), 1)
    sort_index = [arg.tolist().index(i) for i in [0, 1, 2]]
    rotmat = rotmat[sort_index]
    sign = torch.sign(rotmat[torch.arange(3), torch.arange(3)])
    rotmat *= sign
    rotation = Rotate(rotmat.T, device=device)
    points = rotation.transform_points(points)
    points = points.reshape(original_shape)
    if data_type == 'numpy':
        points = points.numpy()
    else:
        points = points
    return points


def texture_uv2vc_t3d(meshes: Meshes):
    device = meshes.device
    vert_uv = meshes.textures.verts_uvs_padded()
    batch_size = vert_uv.shape[0]
    verts_features = []
    num_verts = meshes.verts_padded().shape[1]
    for index in range(batch_size):
        face_uv = vert_uv[index][meshes.textures.faces_uvs_padded()
                                 [index].view(-1)]
        img = meshes.textures._maps_padded[index]
        width, height, _ = img.shape
        face_uv = face_uv * torch.Tensor([width, height]).long().to(device)
        face_uv[:, 0] = torch.clip(face_uv[:, 0], 0, width - 1)
        face_uv[:, 1] = torch.clip(face_uv[:, 1], 0, height - 1)
        face_uv = face_uv.long()
        faces = meshes.faces_padded()
        verts_rgb = torch.zeros(1, num_verts, 3).to(device)
        verts_rgb[:, faces.view(-1)] = img[height - face_uv[:, 1], face_uv[:,
                                                                           0]]
        verts_features.append(verts_rgb)
    verts_features = torch.cat(verts_features)
    meshes = meshes.clone()
    meshes.textures = TexturesVertex(verts_features)
    return meshes


# def texture_uv2vc_o3d(mesh: TriangleMesh):
#     triangle_uvs = np.array(mesh.triangle_uvs)
#     if new_version:
#         im = np.asarray(mesh.textures[0])
#     else:
#         im = np.asarray(mesh.texture)

#     height, width, _ = im.shape
#     triangle_uvs = triangle_uvs * np.array([width, height])
#     triangle_uvs = triangle_uvs.astype(np.int32)
#     triangle_uvs[:, 0] = np.clip(triangle_uvs[:, 0], 0, width - 1)
#     triangle_uvs[:, 1] = np.clip(triangle_uvs[:, 1], 0, height - 1)
#     verts = np.asarray(mesh.vertices)
#     num_verts = verts.shape[0]
#     verts_color = np.zeros((num_verts, 3))
#     triangles = np.asarray(mesh.triangles)
#     verts_color[triangles.reshape(-1)] = im[triangle_uvs[:, 1],
#                                             triangle_uvs[:, 0]] / 255
#     mesh_out = TriangleMesh(vertices=mesh.vertices, triangles=mesh.triangles)
#     mesh_out.vertex_colors = vec3d(verts_color)
#     return mesh_out

# def refine_uv_triangle_mesh(triangle_mesh):
#     if len(triangle_mesh.textures) > 1:
#         mesh_o3d = o3d.geometry.TriangleMesh(
#             vertices=triangle_mesh.vertices, triangles=triangle_mesh.triangles)
#         mesh_o3d.triangle_uvs = triangle_mesh.triangle_uvs
#         mesh_o3d.textures = [triangle_mesh.textures[1]]
#         return mesh_o3d
#     else:
#         return triangle_mesh

# def t3d_to_o3d_mesh(
#     meshes: Meshes,
#     include_textures: bool = True,
# ) -> List[TriangleMesh]:
#     """
#     Convert pytorch3d Meshes to open3d TriangleMesh.
#     Since open3d 0.9.0.0 doesn't support batch meshes, we only feed single
#     `Meshes` of batch N. Will return a list(N) of `TriangleMesh`.

#     Args:
#         meshes (Meshes): batched `Meshes`.
#             Defaults to None.
#         include_textures (bool, optional): whether contain textures.
#             Defaults to False.
#     Returns:
#         List[TriangleMesh]: return a list of open3d `TriangleMesh`.
#     """
#     meshes_o3d = []
#     if meshes is not None:
#         vertices = meshes.verts_padded().clone().detach().cpu().numpy()
#         faces = meshes.faces_padded().clone().detach().cpu().numpy()
#         textures = meshes.textures.clone().detach()
#         batch_size = len(meshes)
#     else:
#         raise ValueError('The input mesh is None. Please pass right inputs.')
#     for index in range(batch_size):
#         mesh_o3d = TriangleMesh(
#             vertices=vec3d(vertices[index]), triangles=vec3i(faces[index]))
#         if include_textures:
#             if isinstance(textures, TexturesVertex):
#                 mesh_o3d.vertex_colors = vec3d(
#                     textures.verts_features_padded()
#                     [index].detach().cpu().numpy())
#             elif isinstance(textures, TexturesUV):
#                 vert_uv = textures.verts_uvs_padded()[index]
#                 face_uv = vert_uv[textures.faces_uvs_padded()[index].view(-1)]

#                 img = textures._maps_padded.cpu().numpy()[index]
#                 img = (img * 255).astype(np.uint8)
#                 img = cv2.flip(img, 0)
#                 if new_version:
#                     mesh_o3d.textures = [o3d.geometry.Image(img)]
#                     mesh_o3d.triangle_uvs = vec2d(
#                         face_uv.detach().cpu().numpy())
#                 else:
#                     mesh_o3d.triangle_uvs = list(
#                         face_uv.detach().cpu().numpy())
#                     mesh_o3d.texture = o3d.geometry.Image(img)
#             elif textures is None:
#                 warnings.warn('Cannot load textures from original mesh.')
#         meshes_o3d.append(mesh_o3d)
#     return meshes_o3d

# def o3d_to_t3d_mesh(meshes: Optional[Union[List[TriangleMesh],
#                                            TriangleMesh]] = None,
#                     include_textures: bool = True) -> Meshes:
#     """
#     Convert open3d TriangleMesh to pytorch3d Meshes .
#     Args:
#         meshes (Optional[Union[List[TriangleMesh], TriangleMesh]], optional):
#             [description]. Defaults to None.
#         include_textures (bool, optional): [description]. Defaults to True.

#     Returns:
#         Meshes: [description]
#     """

#     if not isinstance(meshes, list):
#         meshes = [meshes]
#     vertices = [torch.Tensor(np.asarray(mesh.vertices)) for mesh in meshes]

#     vertices = list_to_padded(vertices, pad_value=0.0)
#     faces = [torch.Tensor(np.asarray(mesh.triangles)) for mesh in meshes]
#     faces = list_to_padded(faces, pad_value=-1.0)
#     if include_textures:
#         has_vertex_colors = meshes[0].has_vertex_colors()
#         if new_version:
#             has_textures = meshes[0].has_textures()
#         else:
#             has_textures = meshes[0].has_texture()
#         if has_vertex_colors:
#             features = [
#                 torch.Tensor(np.asarray(mesh.vertex_colors)) for mesh in meshes
#             ]

#             features = list_to_padded(features, pad_value=0.0)
#             textures = TexturesVertex(verts_features=features)
#         elif has_textures:
#             if new_version:
#                 maps = [
#                     torch.Tensor(
#                         np.asarray(mesh.textures[0]).astype(np.float32))
#                     for mesh in meshes
#                 ]
#             else:
#                 maps = [
#                     torch.Tensor(np.asarray(mesh.texture).astype(np.float32))
#                     for mesh in meshes
#                 ]
#             maps = list_to_padded(maps, pad_size=0) / 255.0
#             faces_uvs = []
#             verts_uvs = []
#             for mesh in meshes:
#                 faces_uv = np.asarray(mesh.triangles)
#                 verts_uv = np.zeros((vertices.shape[1], 2))
#                 verts_uv[faces_uv.reshape(-1)] = np.asarray(mesh.triangle_uvs)
#                 faces_uvs.append(torch.Tensor(faces_uv))
#                 verts_uvs.append(torch.Tensor(verts_uv))
#             faces_uvs = list_to_padded(faces_uvs, pad_value=0.0)
#             verts_uvs = list_to_padded(verts_uvs, pad_value=0.0)
#             textures = TexturesUV(
#                 maps=maps, faces_uvs=faces_uvs, verts_uvs=verts_uvs)
#         else:
#             warnings.warn('Cannot load textures from original mesh.')
#             textures = None
#     else:
#         textures = None
#     meshes_t3d = Meshes(verts=vertices, faces=faces, textures=textures)
#     return meshes_t3d

# def join_batch_axis_as_scene(meshes: Meshes, size=1.0, R=None, T=None):
#     axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
#     axis_mesh = o3d_to_t3d_mesh(axis_mesh)
#     batch_size = len(meshes)
#     axis_mesh = axis_mesh.extend(batch_size)
#     meshes_final = join_batch_meshes_as_scene([meshes, axis_mesh])
#     return meshes_final

# def join_batch_bbox_as_scene(
#         meshes: Meshes,
#         bbox: Optional[Union[torch.Tensor, np.ndarray]] = None) -> Meshes:
#     """
#     Join a batch of mesh with bbox. `bbox` would be generated by function
#     get_oriented_bbox if not provided.

#     Args:
#         meshes (Meshes): `Meshes` which has batch size N.
#         bbox (Optional[Union[torch.Tensor, np.ndarray]], optional):
#             Tensor or array which shape should be (N, 8, 3).
#             If shape is (8, 3), will be extended to (N, 8, 3).

#     Returns:
#         Meshes: a batch of `Meshes` joined with bbox.
#     """
#     if isinstance(bbox, np.ndarray):
#         bbox = torch.Tensor(bbox)
#     elif bbox is None:
#         bbox = get_oriented_bbox(meshes)
#     radius = min(
#         float(
#             torch.min(
#                 torch.sqrt(bbox[:, 0]**2 + bbox[:, 1]**2 + bbox[:, 2]**2)) /
#             20), 0.1)

#     def create_line_mesh(point1: Union[torch.Tensor, np.ndarray],
#                          point2: Union[torch.Tensor,
#                                        np.ndarray], radius: float):
#         diff = point1 - point2
#         length = float(
#             torch.sqrt(diff[:, 0]**2 + diff[:, 1]**2 + diff[:, 2]**2))
#         line = o3d.TriangleMesh.create_cylinder(
#             radius=radius, height=length, resolution=20, split=4)
#         return o3d_to_t3d_mesh(line)

#     line_meshes = []
#     edge_indexs = [[0, 1]]
#     for edge in edge_indexs:
#         line_meshes.append(
#             create_line_mesh(bbox[edge[0]], bbox[edge[1]], radius))
#     line_meshes = join_meshes_as_scene(line_meshes)
#     batch_size = len(meshes)
#     return join_batch_meshes_as_scene([line_meshes.extend(batch_size), meshes])


def uv_to_uvmap(meshes, resolution):
    device = meshes.device
    textures = meshes.textures
    assert isinstance(textures, TexturesUV)
    faces_uv = textures.faces_uvs_padded()[0]
    verts_uv = textures.verts_uvs_padded()[0].float()

    faces = meshes.faces_padded()[0]
    verts = meshes.verts_padded()[0].float()
    texture_map = textures.maps_padded()[0].float()
    # h, w, _ = texture_map.shape
    resolution = texture_map.shape[:2] if resolution is None else resolution
    h, w = resolution

    uv_map = torch.ones(h, w, 3).float().to(device) * verts.min()
    verts_uv_pixel = (verts_uv * torch.tensor([h, w]).to(device).view(1, 2) -
                      0.5).float()
    verts_uv_pixel[:, 1] = h - verts_uv_pixel[:, 1]
    verts_uv_pixel = torch.cat([
        torch.clip(verts_uv_pixel[:, 0:1], min=0, max=h),
        torch.clip(verts_uv_pixel[:, 1:2], min=0, max=w)
    ], 1)

    faces_uv_pixel = verts_uv_pixel[faces_uv.view(-1)].view(-1, 3, 2)
    x = torch.linspace(0, w - 1, w).view(1, -1).repeat(h,
                                                       1).float().unsqueeze(-1)
    y = torch.linspace(0, h - 1, h).view(-1, 1).repeat(1,
                                                       w).float().unsqueeze(-1)
    mesh_grid = torch.cat([x, y], -1).to(device)

    def triangle_area(batch_triangle):
        # B * 3 * 2
        #  S=1/4 * sqrt((a+b+c)(a+b-c)(a+c-b)(b+c-a))
        batch_triangle = batch_triangle.view(-1, 3, 2)
        p1 = batch_triangle[:, 0]
        p2 = batch_triangle[:, 1]
        p3 = batch_triangle[:, 2]
        a = torch.sqrt((p1[:, 0] - p2[:, 0])**2 + (p1[:, 1] - p2[:, 1])**2)
        b = torch.sqrt((p2[:, 0] - p3[:, 0])**2 + (p2[:, 1] - p3[:, 1])**2)
        c = torch.sqrt((p3[:, 0] - p1[:, 0])**2 + (p3[:, 1] - p1[:, 1])**2)
        S = 1 / 4 * torch.sqrt(
            (a + b + c) * (a + b - c) * (a + c - b) * (b + c - a))
        return S

    for triangle_idx in trange(faces_uv_pixel.shape[0]):
        # triangle_coord = faces_uv_pixel[triangle_idx]
        triangle_coord_pixel = faces_uv_pixel[triangle_idx]
        mesh_grid_ = mesh_grid[triangle_coord_pixel[:, 1].min().floor().long(
        ):triangle_coord_pixel[:, 1].max().ceil().long(),
                               triangle_coord_pixel[:, 0].min().floor().long(
                               ):triangle_coord_pixel[:,
                                                      0].max().ceil().long()]
        uv_map_ = uv_map[triangle_coord_pixel[:, 1].min().floor().long(
        ):triangle_coord_pixel[:, 1].max().ceil().long(),
                         triangle_coord_pixel[:, 0].min().floor().long(
                         ):triangle_coord_pixel[:, 0].max().ceil().long()]
        temp_h, temp_w, _ = mesh_grid_.shape

        triangle_xyz = verts[faces[triangle_idx]]
        S = triangle_area(triangle_coord_pixel)
        P = mesh_grid_.reshape(-1, 1, 2)

        batch = P.shape[0]

        A = triangle_coord_pixel[0].view(1, 1, 2).repeat(batch, 1, 1)
        B = triangle_coord_pixel[1].view(1, 1, 2).repeat(batch, 1, 1)
        C = triangle_coord_pixel[2].view(1, 1, 2).repeat(batch, 1, 1)
        S_PBC = triangle_area(torch.cat([P, B, C], 1))
        S_PAC = triangle_area(torch.cat([P, A, C], 1))
        S_PAB = triangle_area(torch.cat([P, A, B], 1))
        barycentric_coord = torch.cat([
            S_PBC.view(-1, 1) / S,
            S_PAC.view(-1, 1) / S,
            S_PAB.view(-1, 1) / S
        ], -1).view(temp_h, temp_w, 3)
        eps = 1e-2
        inside_triangle_index = torch.where(
            torch.abs(barycentric_coord.sum(-1) - 1) < eps)
        barycentric_coord_ = barycentric_coord[inside_triangle_index]
        uv_map_[inside_triangle_index] = (barycentric_coord_.view(
            -1, 1, 3) @ triangle_xyz.view(1, 3, 3)).view(-1, 3)
    return uv_map


def uv_map_to_meshuv(uv_map, meshes, texture_map):
    faces = meshes.faces_padded()[0]
    verts = meshes.verts_padded()[0]
    num_verts = verts.shape[0]
    verts_uv_pixel = torch.zeros(num_verts, 2).long()
    faces_uv = faces
    h, w, _ = uv_map.shape

    def distance(vec1, vec2):
        vec1 = vec1.view(-1, 3)
        vec2 = vec2.view(-1, 3)
        diff = vec1 - vec2
        L = diff[:, 0]**2 + diff[:, 1]**2 + diff[:, 2]**2
        return L

    for vert_idx in trange(num_verts):
        coord = torch.argmin(
            distance(uv_map, verts[vert_idx].view(1, 1, 3)).view(h, w))
        line = coord // h
        row = coord - w * line
        verts_uv_pixel[vert_idx, 0] = row + 0.5
        verts_uv_pixel[vert_idx, 1] = line + 0.5
    verts_uv = verts_uv_pixel / torch.Tensor([w, h]).view(1, 2)
    verts_uv = torch.clip(verts_uv, min=0.0, max=1.0)
    textures = TexturesUV(verts_uvs=verts_uv[None],
                          faces_uvs=faces_uv[None],
                          maps=texture_map[None])
    meshes.textures = textures
    return meshes


# def save_meshes_as_objs(obj_list, meshes: Union[Meshes, TriangleMesh,
#                                                 List[TriangleMesh]]):
#     if not isinstance(obj_list, list):
#         obj_list = [obj_list]
#     if isinstance(meshes, TriangleMesh):
#         meshes = [meshes]
#     if isinstance(meshes, Meshes):
#         if isinstance(meshes.textures, TexturesVertex):
#             meshes = t3d_to_o3d_mesh(meshes)
#     assert len(obj_list) >= len(meshes)
#     if isinstance(meshes, Meshes):

#         if isinstance(meshes.textures, TexturesUV):
#             verts_uvs = meshes.textures.verts_uvs_padded()
#             faces_uvs = meshes.textures.faces_uvs_padded()
#             texture_maps = meshes.textures.maps_padded()
#         else:
#             verts_uvs = None
#             faces_uvs = None
#         verts = meshes.verts_padded()
#         faces = meshes.faces_padded()
#         for index in range(len(meshes)):
#             prepare_output_path(
#                 obj_list[index], allowed_suffix=['.obj'], path_type='file')
#             save_obj(
#                 obj_list[index],
#                 verts=verts[index],
#                 faces=faces[index],
#                 faces_uvs=faces_uvs[index],
#                 verts_uvs=verts_uvs[index],
#                 texture_map=texture_maps[index])
#     else:
#         for index in range(len(meshes)):
#             o3d.io.write_triangle_mesh(
#                 obj_list[index], meshes[index], write_ascii=True)
