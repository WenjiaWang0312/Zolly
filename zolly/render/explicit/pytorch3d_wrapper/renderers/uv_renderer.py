import warnings
import torch
import numpy as np
import torch.nn.functional as F
from typing import Iterable, Optional, Tuple, Union, List

from pytorch3d.io.obj_io import load_objs_as_meshes
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer.mesh import TexturesUV
from pytorch3d.renderer.mesh.rasterizer import (
    MeshRasterizer,
    RasterizationSettings,
)
from pytorch3d.structures import Meshes
from pytorch3d.structures.utils import padded_to_packed

from mmhuman3d.core.conventions.segmentation import body_segmentation
from mmhuman3d.utils.mesh_utils import join_batch_meshes_as_scene
from mmhuman3d.utils.path_utils import check_path_suffix
from zolly.cameras.pytorch3d_wrapper.cameras import (
    FoVOrthographicCameras,
    NewCamerasBase,
)
from .base_renderer import BaseRenderer
from .utils import array2tensor, rgb2bgr


class UVRenderer(BaseRenderer):
    """Renderer for SMPL(x) UV map."""
    shader_type = 'IUVShader'

    def __init__(
        self,
        resolution: Tuple[int] = 1024,
        model_type: Optional[str] = 'smpl',
        uv_param_path: Optional[str] = None,
        obj_path: Optional[str] = None,
        device: Union[torch.device, str] = 'cpu',
        output_path: Optional[str] = None,
        out_img_format: str = '%06d.png',
        bound: tuple = (0, 1),
        **kwargs
        # TODO: Solved the sample bug when the resolution is too small.
        # set threshold_size is just a temporary solution.

        # TODO: add smplx_uv.npz and eval the warping & sampling of smplx
        # model.
    ):
        num_verts = {'smpl': 6890, 'smplx': 10475}
        self.model_type = model_type
        self.NUM_VERTS = num_verts[model_type]
        self.device = device
        self.resolution = (int(resolution), int(resolution)) if isinstance(
            resolution, (int, float)) else resolution
        self.uv_param_path = uv_param_path
        self.obj_path = obj_path
        self.bound = bound
        if uv_param_path is not None:
            check_path_suffix(uv_param_path, allowed_suffix=['npz'])
            param_dict = dict(np.load(uv_param_path))

            verts_uv = torch.Tensor(param_dict['verts_uv'])

            verts_u, verts_v = torch.unbind(verts_uv, -1)
            verts_v_ = 1 - verts_u.unsqueeze(-1)
            verts_u_ = verts_v.unsqueeze(-1)

            self.verts_uv = torch.cat([verts_u_, verts_v_], -1).to(self.device)
            self.faces_uv = torch.LongTensor(param_dict['faces_uv']).to(
                self.device)
            self.faces_uv[12340, 2] = 4188
            self.NUM_VT = self.verts_uv.shape[0]

            self.faces_tensor = torch.LongTensor(param_dict['faces'].astype(
                np.int64)).to(self.device)
            self.num_faces = self.faces_uv.shape[0]
            self.vt2v = torch.Tensor(param_dict['vt2v']).to(self.device).long()

        elif obj_path is not None:
            check_path_suffix(obj_path, allowed_suffix=['obj'])
            mesh_template = load_objs_as_meshes([obj_path])
            self.faces_uv = mesh_template.textures.faces_uvs_padded()[0].to(
                self.device)
            self.verts_uv = mesh_template.textures.verts_uvs_padded()[0].to(
                self.device)
            self.NUM_VT = self.verts_uv.shape[0]
            self.faces_tensor = mesh_template.faces_padded()[0].to(self.device)
            self.num_faces = self.faces_uv.shape[0]
        self.update_fragments()
        self.update_face_uv_pixel()

        super().__init__(resolution=resolution,
                         device=device,
                         output_path=output_path,
                         obj_path=None,
                         out_img_format=out_img_format,
                         **kwargs)

    def to(self, device):
        super().to(device)
        for k in dir(self):
            if isinstance(getattr(self, k), (torch.Tensor)):
                setattr(self, k, getattr(self, k).to(device))
        return self

    def update_fragments(self):
        """Update pix_to_face, bary_coords."""
        rasterizer = MeshRasterizer(cameras=FoVOrthographicCameras(
            min_x=1, max_x=0, max_y=1, min_y=0, device=self.device),
                                    raster_settings=RasterizationSettings(
                                        blur_radius=1e-8,
                                        image_size=self.resolution,
                                        faces_per_pixel=1,
                                        perspective_correct=False,
                                    )).to(self.device)
        verts_uv = torch.cat([
            self.verts_uv[None],
            torch.ones(1, self.NUM_VT, 1).to(self.device)
        ], -1)

        fragments = rasterizer(
            Meshes(verts=verts_uv, faces=self.faces_uv[None]))
        self.pix_to_face = fragments.pix_to_face[0, ..., 0]
        self.bary_coords = fragments.bary_coords[0, ..., 0, :]
        self.mask = (self.pix_to_face >= 0).long()

    def get_warped_part_seg(self, ):
        """Update pix_to_face, bary_coords."""

        faces = self.faces_uv  # (F, 3)
        vt2v = self.vt2v
        colors = torch.zeros(1, self.NUM_VERTS, 1).to(self.device)
        body_segger = body_segmentation(self.model_type)
        for i, k in enumerate(body_segger.keys()):
            colors[:, body_segger[k]] = i + 1

        verts_class_warped = colors[:, vt2v]
        verts_class_warped = padded_to_packed(verts_class_warped)
        faces_class = verts_class_warped[faces]

        bary_coords = self.bary_coords[None].unsqueeze(-2)

        _, idx = torch.max(bary_coords, -1)
        mask = torch.arange(bary_coords.size(-1)).reshape(1, 1, -1).to(
            self.device) == idx.unsqueeze(-1)
        bary_coords *= 0
        bary_coords[mask] = 1

        pix_to_face = self.pix_to_face.unsqueeze(0).unsqueeze(-1)

        part_seg_map = interpolate_face_attributes(
            pix_to_face=pix_to_face,
            barycentric_coords=bary_coords,
            face_attributes=faces_class)
        return part_seg_map[0, :, :, 0, 0].int()

    def update_face_uv_pixel(self):
        #     """Move the pixels lie on the edges inside the mask, then refine the
        #     rest points by searching the nearest pixel in the faces it should be
        #     in."""
        H, W = self.resolution
        # device = self.device
        cameras = FoVOrthographicCameras(min_x=1,
                                         max_x=0,
                                         max_y=1,
                                         min_y=0,
                                         device=self.device)
        verts_uv = torch.cat([
            self.verts_uv[None],
            torch.ones(1, self.NUM_VT, 1).to(self.device)
        ], -1)

        verts_uv_pixel = cameras.transform_points_screen(
            verts_uv, image_size=self.resolution).round().long()[0, ..., :2]
        verts_uv_pixel[..., 0] = torch.clip(verts_uv_pixel[..., 0],
                                            min=0,
                                            max=W - 1)
        verts_uv_pixel[..., 1] = torch.clip(verts_uv_pixel[..., 1],
                                            min=0,
                                            max=H - 1)
        verts_uv_pixel = verts_uv_pixel.long()
        face_uv_pixel = verts_uv_pixel[self.faces_uv]

        face_uv_pixel = face_uv_pixel.long()
        self.face_uv_pixel = face_uv_pixel
        self.face_uv_coord = face_uv_pixel.float() / H

    def render_multi_person(self,
                            meshes: List[Meshes] = None,
                            cameras: Optional[NewCamerasBase] = None,
                            indexes: Optional[Iterable[int]] = None,
                            backgrounds: Optional[torch.Tensor] = None,
                            **kwargs):
        # from mmhuman3d.utils.mesh_utils import join_batch_meshes_as_scene

        num_person = len(meshes)
        meshes = join_batch_meshes_as_scene(meshes)

        meshes = meshes.to(self.device)
        self._update_resolution(cameras, **kwargs)

        fragments = self.rasterizer(meshes_world=meshes, cameras=cameras)
        verts_iuv = self.sample_iuv().repeat(len(meshes), num_person, 1)
        verts_iuv = verts_iuv * (self.bound[1] - self.bound[0]) + self.bound[0]
        iuv_image = self.shader(
            fragments,
            meshes,
            verts_iuv,
        )
        return iuv_image

    def _update_resolution(self, cameras, **kwargs):
        super()._update_resolution(cameras, **kwargs)
        self.update_face_uv_pixel()
        self.update_fragments()

    def forward(self,
                meshes: Optional[Meshes] = None,
                cameras: Optional[NewCamerasBase] = None,
                indexes: Optional[Iterable[int]] = None,
                backgrounds: Optional[torch.Tensor] = None,
                **kwargs):
        """Render depth map.

        Args:
            meshes (Optional[Meshes], optional): meshes to be rendered.
                Defaults to None.
            cameras (Optional[NewCamerasBase], optional): cameras for rendering.
                Defaults to None.
            indexes (Optional[Iterable[int]], optional): indexes for the
                images.
                Defaults to None.
            backgrounds (Optional[torch.Tensor], optional): background images.
                Defaults to None.

        Returns:
            Union[torch.Tensor, None]: return tensor or None.
        """
        meshes = meshes.to(self.device)
        self._update_resolution(cameras, **kwargs)

        fragments = self.rasterizer(meshes_world=meshes, cameras=cameras)
        verts_iuv = self.sample_iuv().repeat_interleave(len(meshes), 0)
        verts_iuv = verts_iuv * (self.bound[1] - self.bound[0]) + self.bound[0]
        iuv_image = self.shader(
            fragments,
            meshes,
            verts_iuv,
        )
        return iuv_image

    def tensor2rgb(self, ):
        pass

    def sample_iuv(self, resolution=None):
        if resolution is None:
            H, W = self.resolution
        else:
            H, W = resolution
        h_grid = torch.linspace(0, 1, H).view(-1, 1).repeat(1, W)
        v_grid = torch.linspace(0, 1, W).repeat(H, 1)

        mesh_grid = torch.cat((v_grid.unsqueeze(2), h_grid.unsqueeze(2)),
                              dim=2)[None].to(self.device)
        vertex_uv = self.vertex_resample(mesh_grid)  # 1, 6890, 2
        vertex_i = torch.ones_like(vertex_uv)[..., 0:1]
        vertex_iuv = torch.cat([vertex_i, vertex_uv], -1)
        return vertex_iuv

    def render_map(self,
                   verts_attr: Optional[torch.Tensor],
                   resolution: Optional[Iterable[int]] = None) -> torch.Tensor:
        """Interpolate the vertex attributes to a map.

        Args:
            verts_attr (Optional[torch.Tensor]): shape should be (N, V, C),
                required.
            resolution (Optional[Iterable[int]], optional): resolution to
                override self.resolution. If None, will use self.resolution.
                Defaults to None.

        Returns:
            torch.Tensor: interpolated maps of (N, H, W, C)
        """
        if verts_attr.ndim == 2:
            verts_attr = verts_attr[None]
        if resolution is not None and resolution != self.resolution:
            self.resolution = resolution
            self.update_fragments()
            self.update_face_uv_pixel()

        bary_coords = self.bary_coords
        pix_to_face = self.pix_to_face

        N, V, C = verts_attr.shape
        assert V == self.NUM_VERTS
        verts_attr = verts_attr.view(N * V, C).to(self.device)
        offset_idx = torch.arange(0, N).long() * (self.NUM_VERTS - 1)
        faces_packed = self.faces_tensor[None].repeat(
            N, 1, 1) + offset_idx.view(-1, 1, 1).to(self.device)
        faces_packed = faces_packed.view(-1, 3)
        face_attr = verts_attr[faces_packed]
        assert face_attr.shape == (N * self.num_faces, 3, C)
        pix_to_face = self.pix_to_face.unsqueeze(0).repeat(N, 1,
                                                           1).unsqueeze(-1)
        bary_coords = self.bary_coords[None].repeat(N, 1, 1, 1).unsqueeze(-2)
        maps_padded = interpolate_face_attributes(
            pix_to_face=pix_to_face.to(self.device),
            barycentric_coords=bary_coords.to(self.device),
            face_attributes=face_attr.to(self.device),
        ).squeeze(-2)
        return maps_padded

    def render_normal_map(self,
                          meshes: Meshes = None,
                          vertices: torch.Tensor = None,
                          resolution: Optional[Iterable[int]] = None,
                          cameras: NewCamerasBase = None) -> torch.Tensor:
        """Interpolate verts normals to a normal map.

        Args:
            meshes (Meshes): input smpl mesh.
                Will override vertices if both not None.
                Defaults to None.
            vertices (torch.Tensor, optional):
                smpl vertices. Defaults to None.
            resolution (Optional[Iterable[int]], optional): resolution to
                override self.resolution. If None, will use self.resolution.
                Defaults to None.
            cameras (NewCamerasBase, optional):
                cameras to see the mesh.
                Defaults to None.
        Returns:
            torch.Tensor: Normal map of shape (N, H, W, 3)
        """
        if meshes is not None:
            verts_normals = meshes.verts_normals_padded()
        elif meshes is None and vertices is not None:
            meshes = Meshes(verts=vertices,
                            faces=self.faces_tensor[None].repeat(
                                vertices.shape[0], 1, 1))
            verts_normals = meshes.verts_normals_padded()
        else:
            raise ValueError('No valid input.')
        verts_normals = meshes.verts_normals_padded()
        if cameras:
            verts_normals = cameras.get_world_to_view_transform(
            ).transform_normals(verts_normals)
        normal_map = self.render_map(verts_attr=verts_normals,
                                     resolution=resolution)
        return normal_map

    def render_uvd_map(self,
                       meshes: Meshes = None,
                       vertices: torch.Tensor = None,
                       resolution: Optional[Iterable[int]] = None,
                       cameras: NewCamerasBase = None) -> torch.Tensor:
        """Interpolate the verts xyz value to a uvd map.

        Args:
            meshes (Meshes): input smpl mesh.
                Defaults to None.
            vertices (torch.Tensor, optional):
                smpl vertices. Will override meshes if both not None.
                Defaults to None.
            resolution (Optional[Iterable[int]], optional): resolution to
                override self.resolution. If None, will use self.resolution.
                Defaults to None.
            cameras (NewCamerasBase, optional):
                cameras to see the mesh.
                Defaults to None.

        Returns:
            torch.Tensor: UVD map of shape (N, H, W, 3)
        """
        if vertices is not None:
            verts_uvd = vertices
        elif vertices is None and meshes is not None:
            verts_uvd = meshes.verts_padded()
        else:
            raise ValueError('No valid input.')
        if cameras:
            verts_uvd = cameras.get_world_to_view_transform(
            ).transform_normals(verts_uvd)
        uvd_map = self.render_map(verts_attr=verts_uvd, resolution=resolution)
        return uvd_map

    def vertex_resample(
        self,
        maps_padded: torch.Tensor,
        h_flip: bool = False,
        use_float: bool = False,
    ) -> torch.Tensor:
        """Resample the vertex attributes from a map.

        Args:
            maps_padded (torch.Tensor): shape should be (N, H, W, C). Required.
            h_flip (bool, optional): whether flip horizontally.
                Defaults to False.

        Returns:
            torch.Tensor: resampled vertex attributes. Shape will be (N, V, C)
        """
        if maps_padded.ndim == 3:
            maps_padded = maps_padded[None]

        if h_flip:
            maps_padded = torch.flip(maps_padded, dims=[2])
        N, H, W, C = maps_padded.shape

        # if H < self.threshold_size or W < self.threshold_size:
        #     maps_padded = F.interpolate(
        #         maps_padded.permute(0, 3, 1, 2),
        #         size=(self.threshold_size, self.threshold_size),
        #         mode='bicubic',
        #         align_corners=False).permute(0, 2, 3, 1)
        #     H, W = self.threshold_size, self.threshold_size
        if (H, W) != self.resolution:
            self.resolution = (H, W)
            self.update_fragments()
            self.update_face_uv_pixel()
        offset_idx = torch.arange(0, N).long() * (self.NUM_VERTS - 1)
        faces_packed = self.faces_tensor[None].repeat(
            N, 1, 1) + offset_idx.view(-1, 1, 1).to(self.device)
        faces_packed = faces_packed.view(-1, 3)

        verts_feature_packed = torch.zeros(N * self.NUM_VERTS,
                                           C).to(self.device)
        if use_float:

            faces_uv = self.verts_uv[self.faces_uv]  # F, 3, 2
            faces_uv = faces_uv[None].repeat_interleave(N, 0)  # N, F, 3, 2
            faces_uv = torch.cat([faces_uv[..., 1:2], faces_uv[..., 0:1]], -1)
            verts_feature_packed[faces_packed] = F.grid_sample(
                maps_padded.permute(0, 3, 1, 2),
                faces_uv * 2 - 1,
                align_corners=True).permute(0, 2, 3, 1).reshape(
                    N * self.num_faces, 3, C)  # N, C, F, 3 -> N, F, 3, C
        else:

            face_uv_pixel = self.face_uv_pixel.view(-1, 2)
            verts_feature_packed[
                faces_packed] = maps_padded[:, face_uv_pixel[:, 1],
                                            face_uv_pixel[:, 0]].view(
                                                N * self.num_faces, 3,
                                                C)  # N*F, 3, C
        verts_feature_padded = verts_feature_packed.view(N, self.NUM_VERTS, C)

        return verts_feature_padded

    def wrap_normal(
        self,
        meshes: Meshes,
        normal: torch.Tensor = None,
        normal_map: torch.Tensor = None,
    ) -> Meshes:
        """Warp a normal map or vertex normal to the input meshes.

        Args:
            meshes (Meshes): the input meshes.
            normal (torch.Tensor, optional): vertex normal. Shape should be
                (N, V, 3).
                Defaults to None.
            normal_map (torch.Tensor, optional):
                normal map. Defaults to None.

        Returns:
            Meshes: returned meshes.
        """
        if normal_map is not None and normal is None:
            normal = self.vertex_resample(normal_map)
        elif normal_map is not None and normal is not None:
            normal_map = None
        elif normal_map is None and normal is None:
            warnings.warn('Redundant input, will only take displacement.')
        batch_size = len(meshes)
        if normal.ndim == 2:
            normal = normal[None]
        assert normal.shape[1:] == (self.NUM_VERTS, 3)
        assert normal.shape[0] in [batch_size, 1]

        if normal.shape[0] == 1:
            normal = normal.repeat(batch_size, 1, 1)
        meshes = meshes.clone()

        meshes._set_verts_normals(normal)
        return meshes

    def wrap_displacement(
        self,
        meshes: Meshes,
        displacement: torch.Tensor = None,
        displacement_map: torch.Tensor = None,
    ) -> Meshes:
        """Offset a vertex displacement or displacement_map to the input
        meshes.

        Args:
            meshes (Meshes): the input meshes.
            displacement (torch.Tensor, optional): vertex displacement.
                shape should be (N, V, 3).
                Defaults to None.
            displacement_map (torch.Tensor, optional): displacement_map,
                shape should be (N, H, W, 3).
                Defaults to None.

        Returns:
            Meshes: returned meshes.
        """
        if displacement_map is not None and displacement is None:
            displacement = self.vertex_resample(displacement_map)
        elif displacement_map is not None and displacement is not None:
            displacement_map = None
            warnings.warn('Redundant input, will only take displacement.')
        elif displacement_map is None and displacement is None:
            raise ValueError('No valid input.')
        batch_size = len(meshes)
        if displacement.ndim == 2:
            displacement = displacement[None]
        assert displacement.shape[1] == self.NUM_VERTS
        assert displacement.shape[0] in [batch_size, 1]

        if displacement.shape[0] == 1:
            displacement = displacement.repeat(batch_size, 1, 1)
        C = displacement.shape[-1]
        if C == 1:
            displacement = meshes.verts_normals_padded() * displacement

        displacement = padded_to_packed(displacement)

        meshes = meshes.to(self.device)
        meshes = meshes.offset_verts(displacement)
        return meshes

    def wrap_texture(self,
                     texture_map: torch.Tensor,
                     resolution: Optional[Iterable[int]] = None,
                     mode: Optional[str] = 'bicubic',
                     is_bgr: bool = True) -> Meshes:
        """Wrap a texture map to the input meshes.

        Args:
            texture_map (torch.Tensor): the texture map to be wrapped.
                Shape should be (N, H, W, 3)
            resolution (Optional[Iterable[int]], optional): resolution to
                override self.resolution. If None, will use self.resolution.
                Defaults to None.
            mode (Optional[str], optional): interpolate mode.
                Should be in ['nearest', 'bilinear', 'trilinear', 'bicubic',
                'area'].
                Defaults to 'bicubic'.
            is_bgr (bool, optional): Whether the color channel is BGR.
                Defaults to True.

        Returns:
            Meshes: returned meshes.
        """

        assert texture_map.shape[-1] == 3
        if texture_map.ndim == 3:
            texture_map_padded = texture_map[None]
        elif texture_map.ndim == 4:
            texture_map_padded = texture_map
        else:
            raise ValueError(f'Wrong texture_map shape: {texture_map.shape}.')
        N, H, W, _ = texture_map_padded.shape

        resolution = resolution if resolution is not None else (H, W)

        if resolution != (H, W):
            texture_map_padded = F.interpolate(texture_map_padded.view(
                0, 3, 1, 2),
                                               resolution,
                                               mode=mode).view(0, 2, 3, 1)
        assert texture_map_padded.shape[0] in [N, 1]

        if isinstance(texture_map_padded, np.ndarray):
            texture_map_padded = array2tensor(texture_map_padded)
            is_bgr = True
        if is_bgr:
            texture_map_padded = rgb2bgr(texture_map_padded)

        if texture_map_padded.shape[0] == 1:
            texture_map_padded = texture_map_padded.repeat(N, 1, 1, 1)

        faces_uvs = self.faces_uv[None].repeat(N, 1, 1)
        verts_uvs = self.verts_uv[None].repeat(N, 1, 1)
        textures = TexturesUV(faces_uvs=faces_uvs,
                              verts_uvs=verts_uvs,
                              maps=texture_map_padded)
        return textures

    def inverse_wrap(self, iuv_image, features):
        # iuv_image B, 3, H, W
        # feature B, C, H, W
        iuv_image = (iuv_image - self.bound[0]) / (self.bound[1] -
                                                   self.bound[0])
        iuv_image = iuv_image * 2 - 1
        device = iuv_image.device
        B, _, uv_h, uv_w = iuv_image.shape
        # h_grid = torch.linspace(0, 1, H).view(-1, 1).repeat(1, W)

        # v_grid = torch.linspace(0, 1, W).repeat(H, 1)

        h_grid = torch.linspace(-1, 1, uv_h).view(-1, 1).repeat(1, uv_w)
        v_grid = torch.linspace(-1, 1, uv_w).repeat(uv_h, 1)

        xy = torch.cat(
            (v_grid.unsqueeze(2), h_grid.unsqueeze(2)),
            dim=2)[None].repeat_interleave(B, 0).to(device)  # B, H, W, 2
        mesh_grid = xy.clone()

        # xy = mesh_grid.clone()
        uv_image = iuv_image[:, 1:]  # B, 2, H, W
        uv = uv_image / 2 + 0.5
        u = (uv[:, 0] * (uv_w - 1)).long()
        v = (uv[:, 1] * (uv_h - 1)).long()
        mask = torch.zeros(B, 1, uv_h, uv_w).to(device)

        index_batch = torch.arange(B).view(B, 1,
                                           1).repeat(1, uv_h,
                                                     uv_w).view(B,
                                                                -1).to(device)
        mesh_grid[index_batch.view(-1),
                  v.view(-1), u.view(-1)] = xy.view(-1, 2)

        mesh_grid = mesh_grid.view(B, uv_h, uv_w, 2)

        mask[index_batch.view(-1), :, v.view(-1), u.view(-1)] = 1

        wraped_features = F.grid_sample(features,
                                        mesh_grid,
                                        align_corners=False,
                                        mode='nearest')
        wraped_features = wraped_features * mask
        return wraped_features


def process_decomr():
    import math
    dic = dict(np.load('data/uv_sampler/paras_h0128_w0128_BF.npz'))
    dic['faces_uv'] = np.zeros((13776, 3)).astype(np.int32)
    for face_index, face in enumerate(dic['faces']):
        v0, v1, v2 = face
        v0_ = np.where(dic['vt2v'] == v0)[0]
        v1_ = np.where(dic['vt2v'] == v1)[0]
        v2_ = np.where(dic['vt2v'] == v2)[0]

        if len(v0_) == 1:
            v0_final = v0_
            v_anchor = v0_
        else:
            v0_final = None

        if len(v1_) == 1:
            v1_final = v1_
            v_anchor = v1_
        else:
            v1_final = None

        if len(v2_) == 1:
            v2_final = v2_
            v_anchor = v2_
        else:
            v2_final = None

        def dist(coord1, coord2):

            dist = math.sqrt((coord1[0] - coord2[0])**2 +
                             (coord1[1] - coord2[1])**2)
            return dist

        anchor_coord = dic['verts_uv'][v_anchor][0]
        if v0_final is None:
            for v in v0_:
                if dist(dic['verts_uv'][v], anchor_coord) <= 0.05:
                    v0_final = v

        if v1_final is None:
            for v in v1_:
                if dist(dic['verts_uv'][v], anchor_coord) <= 0.05:
                    v1_final = v

        if v2_final is None:
            for v in v2_:
                if dist(dic['verts_uv'][v], anchor_coord) <= 0.05:
                    v2_final = v

        if (v0_final is None) or (v1_final is None) or (v2_final is None):

            coords0 = dic['verts_uv'][v0_]
            coords1 = dic['verts_uv'][v1_]
            coords2 = dic['verts_uv'][v2_]
            for i_0, coord0 in enumerate(coords0):
                for i_1, coord1 in enumerate(coords1):
                    if dist(coord0, coord1) < 0.05:
                        for i_2, coord2 in enumerate(coords2):
                            if dist(coord0, coord2) < 0.05:
                                v0_final = v0_[i_0]
                                v1_final = v1_[i_1]
                                v2_final = v0_[i_2]

        dic['faces_uv'][face_index, 0] = v0_final
        dic['faces_uv'][face_index, 1] = v1_final
        dic['faces_uv'][face_index, 2] = v2_final

    np.savez('smpl_uv_decomr.npz', **dic)
