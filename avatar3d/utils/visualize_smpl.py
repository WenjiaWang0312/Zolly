import copy
import os
import os.path as osp
import shutil
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import mmcv
import numpy as np
import torch
import torch.nn as nn
from colormap import Color

import mmhuman3d
from mmhuman3d.core.conventions.segmentation import body_segmentation
from mmhuman3d.utils.demo_utils import get_different_colors
from mmhuman3d.utils.mesh_utils import save_meshes_as_objs, save_meshes_as_plys
from mmhuman3d.utils.path_utils import check_path_suffix, check_input_path, prepare_output_path

from avatar3d.cameras.pytorch3d_wrapper.cameras import (
    NewCamerasBase,
    compute_orbit_cameras,
)
from avatar3d.cameras.builder import build_cameras

from avatar3d.render.explicit.pytorch3d_wrapper import render_runner
from avatar3d.render.explicit.pytorch3d_wrapper.renderers.utils import align_input_to_padded
from avatar3d.structures.meshes.meshes import ParametricMeshes
from avatar3d.render.explicit.pytorch3d_wrapper.renderers.smpl_renderer import SMPLRenderer

from avatar3d.utils.frame_utils import (images_to_array, video_to_array,
                                        video_to_images, resize_array)
from avatar3d.utils.torch_utils import dict2tensor, to_tensor

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def _prepare_background(image_array, frame_list, origin_frames, output_path,
                        start, end, img_format, overwrite, num_frames,
                        read_frames_batch, resolution):
    """Compare among `image_array`, `frame_list` and `origin_frames` and decide
    whether to save the temp background images."""
    if num_frames > 300:
        read_frames_batch = True

    frames_folder = None
    remove_folder = False

    if isinstance(image_array, np.ndarray):

        image_array = torch.Tensor(image_array)

    if image_array is not None:
        if image_array.ndim == 3:
            image_array = image_array[None]
        if image_array.shape[0] == 1:
            image_array = image_array.repeat(num_frames, 1, 1, 1)
        frame_list = None
        origin_frames = None
        image_array = image_array[start:end]

    # check the output path and get the image_array
    if output_path is not None:
        prepare_output_path(output_path=output_path,
                            allowed_suffix=['.mp4', 'gif', 'png', 'jpg', ''],
                            tag='output video',
                            path_type='auto',
                            overwrite=overwrite)
        if image_array is None:
            # choose in frame_list or origin_frames
            # if all None, will use pure white background
            if frame_list is None and origin_frames is None:
                print(
                    'No background provided, will use pure white background.')
            elif frame_list is not None and origin_frames is not None:
                warnings.warn('Redundant input, will only use frame_list.')
                origin_frames = None

            # read the origin frames as array if any.
            if frame_list is None and origin_frames is not None:
                check_input_path(input_path=origin_frames,
                                 allowed_suffix=['.mp4', '.gif', ''],
                                 tag='origin frames',
                                 path_type='auto')
                # if origin_frames is a video, write it as a folder of images
                # if read_frames_batch is True, else read directly as an array.
                if Path(origin_frames).is_file():
                    if read_frames_batch:
                        frames_folder = osp.join(
                            Path(output_path).parent,
                            Path(output_path).name + '_input_temp')
                        os.makedirs(frames_folder, exist_ok=True)
                        video_to_images(origin_frames,
                                        frames_folder,
                                        img_format=img_format,
                                        start=start,
                                        end=end)
                        remove_folder = True
                    else:
                        remove_folder = False
                        frames_folder = None
                        image_array = video_to_array(origin_frames,
                                                     start=start,
                                                     end=end,
                                                     resolution=resolution)
                # if origin_frames is a folder, write it as a folder of images
                # read the folder as an array if read_frames_batch is True
                # else return frames_folder for reading during rendering.
                else:
                    if read_frames_batch:
                        frames_folder = origin_frames
                        remove_folder = False
                        image_array = None
                    else:
                        image_array = images_to_array(origin_frames,
                                                      img_format=img_format,
                                                      start=start,
                                                      end=end,
                                                      resolution=resolution)
                        remove_folder = False
                        frames_folder = origin_frames
            # if frame_list is not None, move the images into a folder
            # read the folder as an array if read_frames_batch is True
            # else return frames_folder for reading during rendering.
            elif frame_list is not None and origin_frames is None:
                frames_folder = osp.join(
                    Path(output_path).parent,
                    Path(output_path).name + '_input_temp')
                os.makedirs(frames_folder, exist_ok=True)
                for frame_idx, frame_path in enumerate(frame_list):
                    if check_path_suffix(
                            path_str=frame_path,
                            allowed_suffix=['.jpg', '.png', '.jpeg']):
                        shutil.copy(
                            frame_path,
                            os.path.join(frames_folder,
                                         '%06d.png' % frame_idx))
                        img_format = '%06d.png'
                if not read_frames_batch:

                    image_array = images_to_array(frames_folder,
                                                  img_format=img_format,
                                                  remove_raw_files=True,
                                                  resolution=resolution)
                    frames_folder = None
                    remove_folder = False
                else:
                    image_array = None
                    remove_folder = True
    return image_array, remove_folder, frames_folder


def _prepare_input_pose(verts, poses, betas, transl, device):
    """Prepare input pose data as tensor and ensure correct temporal slice."""
    if verts is None and poses is None:
        raise ValueError('Please input valid poses or verts.')
    elif (verts is not None) and (poses is not None):
        warnings.warn('Redundant input, will take verts and ignore poses & '
                      'betas & transl.')
        poses = None
        transl = None
        betas = None

    if verts is not None:
        verts = to_tensor(verts, device)
        num_frames = verts.shape[0]
    elif isinstance(poses, dict):
        transl = poses.get('transl', transl)
        betas = poses.get('betas', betas)
        transl = to_tensor(transl, device)
        betas = to_tensor(betas, device)
    elif isinstance(poses, dict):
        poses = dict2tensor(poses, device)
        num_frames = poses['body_pose'].shape[0]
    else:
        poses = to_tensor(poses, device)
        if poses is not None:
            num_frames = poses.shape[0]

    if betas is not None:
        if betas.shape[0] != num_frames:
            times = num_frames // betas.shape[0]
            if betas.ndim == 2:
                betas = betas.repeat(times, 1)[:num_frames]
            elif betas.ndim == 3:
                betas = betas.repeat(times, 1, 1)[:num_frames]
            print(f'betas will be repeated by dim 0 for {times} times.')

    return verts, poses, betas, transl


def _prepare_mesh(poses, betas, transl, verts, start, end, body_model):
    """Prepare the mesh info for rendering."""
    NUM_JOINTS = body_model.NUM_JOINTS
    NUM_BODY_JOINTS = body_model.NUM_BODY_JOINTS
    NUM_DIM = 3 * (NUM_JOINTS + 1)
    body_pose_keys = set(body_model.body_pose_dims.keys())
    joints = None

    if poses is not None:
        if isinstance(poses, dict):
            if not body_pose_keys.issubset(poses):
                raise KeyError(
                    f'{str(poses.keys())}, Please make sure that your '
                    f'input dict has all of {", ".join(body_pose_keys)}')
            num_frames = poses['body_pose'].shape[0]
            _, num_person, _ = poses['body_pose'].view(
                num_frames, -1, NUM_BODY_JOINTS * 3).shape

            full_pose = body_model.dict2tensor(poses)
            full_pose = full_pose[start:end]

        elif isinstance(poses, torch.Tensor):
            if poses.shape[-1] != NUM_DIM:
                raise ValueError(
                    f'Please make sure your poses is {NUM_DIM} dims in'
                    f'the last axis. Your input shape: {poses.shape}')
            poses = poses.view(poses.shape[0], -1, (NUM_JOINTS + 1) * 3)
            num_frames, num_person, _ = poses.shape
            full_pose = poses[start:end]
        else:
            raise ValueError('Wrong pose type, should be `dict` or `tensor`.')

        # multi person check
        if num_person > 1:
            if betas is not None:
                num_betas = betas.shape[-1]
                betas = betas.view(num_frames, -1, num_betas)

                if betas.shape[1] == 1:
                    betas = betas.repeat(1, num_person, 1)
                    warnings.warn(
                        'Only one betas for multi-person, will all be the '
                        'same body shape.')
                elif betas.shape[1] > num_person:
                    betas = betas[:, :num_person]
                    warnings.warn(
                        f'Betas shape exceed, will be sliced as {betas.shape}.'
                    )
                elif betas.shape[1] == num_person:
                    pass
                else:
                    raise ValueError(
                        f'Odd betas shape: {betas.shape}, inconsistent'
                        f'with poses in num_person: {poses.shape}.')
            else:
                warnings.warn('None betas for multi-person, will all be the '
                              'default body shape.')

            if transl is not None:
                transl = transl.view(poses.shape[0], -1, 3)
                if transl.shape[1] == 1:
                    transl = transl.repeat(1, num_person, 1)
                    warnings.warn(
                        'Only one transl for multi-person, will all be the '
                        'same translation.')
                elif transl.shape[1] > num_person:
                    transl = transl[:, :num_person]
                    warnings.warn(f'Transl shape exceed, will be sliced as'
                                  f'{transl.shape}.')
                elif transl.shape[1] == num_person:
                    pass
                else:
                    raise ValueError(
                        f'Odd transl shape: {transl.shape}, inconsistent'
                        f'with poses in num_person: {poses.shape}.')
            else:
                warnings.warn('None transl for multi-person, will all be the '
                              'default translation.')

        # slice the input poses, betas, and transl.
        betas = betas[start:end] if betas is not None else None
        transl = transl[start:end] if transl is not None else None
        pose_dict = body_model.fullpose_to_dict(full_pose=full_pose)
        pose_dict.update(betas=betas, transl=transl)

        # get new num_frames
        num_frames = full_pose.shape[0]

        model_output = body_model(**pose_dict)
        vertices = model_output['vertices']
        joints = model_output['joints']

    elif verts is not None:
        if isinstance(verts, np.ndarray):
            verts = torch.Tensor(verts)
        verts = verts[start:end]

        if verts.ndim == 3:
            joints = torch.einsum('bik,ji->bjk',
                                  [verts, body_model.J_regressor])
        elif verts.ndim == 4:
            joints = torch.einsum('fpik,ji->fpjk',
                                  [verts, body_model.J_regressor])
        num_verts = body_model.NUM_VERTS
        assert verts.shape[-2] == num_verts, 'Wrong input verts shape.'
        num_frames = verts.shape[0]
        vertices = verts.view(num_frames, -1, num_verts, 3)
        num_joints = joints.shape[-2]
        num_person = vertices.shape[1]
        joints = joints.view(num_frames, num_joints * num_person, 3)
    else:
        raise ValueError('Poses and verts are all None.')
    return vertices, joints, num_frames, num_person


def _prepare_colors(palette, render_choice, num_person, num_verts, model_type):
    """Prepare the `color` as a tensor of shape (num_person, num_verts, 3)
    according to `palette`.

    This is to make the identity in video clear.
    """
    if not len(palette) == num_person:
        raise ValueError('Please give the right number of palette.')

    if render_choice == 'silhouette':
        colors = torch.ones(num_person, num_verts, 3)
    elif render_choice == 'part_silhouette':
        colors = torch.zeros(num_person, num_verts, 3)
        body_segger = body_segmentation(model_type)
        for i, k in enumerate(body_segger.keys()):
            colors[:, body_segger[k]] = i + 1
    else:
        if isinstance(palette, torch.Tensor):
            if palette.max() > 1:
                palette = palette / 255.0
            palette = torch.clip(palette, min=0, max=1)
            colors = palette.view(num_person,
                                  3).unsqueeze(1).repeat(1, num_verts, 1)

        elif isinstance(palette, list):
            colors = []
            for person_idx in range(num_person):

                if palette[person_idx] == 'random':
                    color_person = get_different_colors(
                        num_person, int_dtype=False)[person_idx]
                    color_person = torch.FloatTensor(color_person)
                    color_person = torch.clip(color_person * 1.5,
                                              min=0.6,
                                              max=1)
                    color_person = color_person.view(1, 1, 3).repeat(
                        1, num_verts, 1)
                elif palette[person_idx] == 'segmentation':
                    body_segger = body_segmentation(model_type)
                    verts_labels = torch.zeros(num_verts)
                    color_person = torch.ones(1, num_verts, 3)
                    color_part = get_different_colors(len(body_segger),
                                                      int_dtype=False)
                    for part_idx, k in enumerate(body_segger.keys()):
                        index = body_segger[k]
                        verts_labels[index] = part_idx
                        color_person[:, index] = torch.FloatTensor(
                            color_part[part_idx])
                elif palette[person_idx] in Color.color_names:
                    color_person = torch.FloatTensor(
                        Color(palette[person_idx]).rgb).view(1, 1, 3).repeat(
                            1, num_verts, 1)
                else:
                    raise ValueError('Wrong palette string. '
                                     'Please choose in the pre-defined range.')
                colors.append(color_person)
            colors = torch.cat(colors, 0)
            assert colors.shape == (num_person, num_verts, 3)
            # the color passed to renderer will be (num_person, num_verts, 3)
        else:
            raise ValueError(
                'Palette should be tensor, array or list of strs.')
    return colors


def vis_smpl(
        # smpl parameters
        poses: Optional[Union[torch.Tensor, np.ndarray, dict]] = None,
        betas: Optional[Union[torch.Tensor, np.ndarray]] = None,
        transl: Optional[Union[torch.Tensor, np.ndarray]] = None,
        verts: Optional[Union[torch.Tensor, np.ndarray]] = None,
        body_model: Optional[nn.Module] = None,
        # camera parameters
        cameras: NewCamerasBase = None,
        orbit_speed: Union[float, Tuple[float, float]] = 0.0,
        # render choice parameters
        render_choice: Literal['lq', 'mq', 'hq', 'silhouette', 'depth',
                               'normal', 'pointcloud',
                               'part_silhouette'] = 'hq',
        palette: Union[List[str], str, np.ndarray, torch.Tensor] = 'white',
        texture_image: Union[torch.Tensor, np.ndarray] = None,
        resolution: Optional[Union[List[int], Tuple[int, int]]] = None,
        start: int = 0,
        end: Optional[int] = None,
        alpha: float = 1.0,
        no_grad: bool = True,
        batch_size: int = 10,
        device: Union[torch.device, str] = 'cuda',
        # file io parameters
        return_tensor: bool = False,
        output_path: str = None,
        origin_frames: Optional[str] = None,
        frame_list: Optional[List[str]] = None,
        image_array: Optional[Union[np.ndarray, torch.Tensor]] = None,
        img_format: str = '%06d.png',
        overwrite: bool = False,
        mesh_file_path: Optional[str] = None,
        read_frames_batch: bool = False,
        # visualize keypoints
        plot_pcs: bool = False,
        kp3d: Optional[Union[np.ndarray, torch.Tensor]] = None,
        mask: Optional[Union[np.ndarray, List[int]]] = None,
        vis_kp_index: bool = False,
        verbose: bool = False) -> Union[None, torch.Tensor]:
    """Render SMPL or SMPL-X mesh or silhouette into differentiable tensors,
    and export video or images.

    Args:
        # smpl parameters:
        poses (Union[torch.Tensor, np.ndarray, dict]):

            1). `tensor` or `array` and ndim is 2, shape should be
            (frame, 72).

            2). `tensor` or `array` and ndim is 3, shape should be
            (frame, num_person, 72/165). num_person equals 1 means
            single-person.
            Rendering predicted multi-person should feed together with
            multi-person weakperspective cameras. meshes would be computed
            and use an identity intrinsic matrix.

            3). `dict`, standard dict format defined in smplx.body_models.
            will be treated as single-person.

            Lower priority than `verts`.

            Defaults to None.
        betas (Optional[Union[torch.Tensor, np.ndarray]], optional):
            1). ndim is 2, shape should be (frame, 10).

            2). ndim is 3, shape should be (frame, num_person, 10). num_person
            equals 1 means single-person. If poses are multi-person, betas
            should be set to the same person number.

            None will use default betas.

            Defaults to None.
        transl (Optional[Union[torch.Tensor, np.ndarray]], optional):
            translations of smpl(x).

            1). ndim is 2, shape should be (frame, 3).

            2). ndim is 3, shape should be (frame, num_person, 3). num_person
            equals 1 means single-person. If poses are multi-person,
            transl should be set to the same person number.

            Defaults to None.
        verts (Optional[Union[torch.Tensor, np.ndarray]], optional):
            1). ndim is 3, shape should be (frame, num_verts, 3).

            2). ndim is 4, shape should be (frame, num_person, num_verts, 3).
            num_person equals 1 means single-person.

            Higher priority over `poses` & `betas` & `transl`.

            Defaults to None.
        body_model (nn.Module, optional): body_model created from smplx.create.
            Higher priority than `body_model_config`. If `body_model` is not
            None, it will override `body_model_config`.
            Should not both be None.

            Defaults to None.

        cameras (NewCamerasBase): 

        orbit_speed (float, optional): orbit speed for viewing when no `K`
            provided. `float` for only azim speed and Tuple for `azim` and
            `elev`.

        # render choice parameters:

        render_choice (Literal[, optional):
            choose in ['lq', 'mq', 'hq', 'silhouette', 'depth', 'normal',
            'pointcloud', 'part_silhouette'] .

            `lq`, `mq`, `hq` would output (frame, h, w, 4) FloatTensor.

            `lq` means low quality, `mq` means medium quality,
            h`q means high quality.

            `silhouette` would output (frame, h, w) soft binary FloatTensor.

            `part_silhouette` would output (frame, h, w, 1) LongTensor.

            Every pixel stores a class index.

            `depth` will output a depth map of (frame, h, w, 1) FloatTensor
            and 'normal' will output a normal map of (frame, h, w, 1).

            `pointcloud` will output a (frame, h, w, 4) FloatTensor.

            Defaults to 'mq'.
        palette (Union[List[str], str, np.ndarray], optional):
            color theme str or list of color str or `array`.

            1). If use str to represent the color,
            should choose in ['segmentation', 'random'] or color from
            Colormap https://en.wikipedia.org/wiki/X11_color_names.
            If choose 'segmentation', will get a color for each part.

            2). If you have multi-person, better give a list of str or all
            will be in the same color.

            3). If you want to define your specific color, use an `array`
            of shape (3,) for single person and (N, 3) for multiple persons.

            If (3,) for multiple persons, all will be in the same color.

            Your `array` should be in range [0, 255] for 8 bit color.

            Defaults to 'white'.

        texture_image (Union[torch.Tensor, np.ndarray], optional):
            Texture image to be wrapped on the smpl mesh. If not None,
            the `palette` will be ignored, and the `body_model` is required
            to have `uv_param_path`.
            Should pass list or tensor of shape (num_person, H, W, 3).
            The color channel should be `RGB`.

            Defaults to None.

        resolution (Union[Iterable[int], int], optional):
            1). If iterable, should be (height, width) of output images.

            2). If int, would be taken as (resolution, resolution).

            Defaults to (1024, 1024).

            This will influence the overlay results when render with
            backgrounds. The output video will be rendered following the
            size of background images and finally resized to resolution.
        start (int, optional): start frame index. Defaults to 0.

        end (int, optional): end frame index. Exclusive.
                Could be positive int or negative int or None.
                None represents include all the frames.

            Defaults to None.
        alpha (float, optional): Transparency of the mesh.
            Range in [0.0, 1.0]

            Defaults to 1.0.
        no_grad (bool, optional): Set to True if do not need differentiable
            render.

            Defaults to False.
        batch_size (int, optional):  Batch size for render.
            Related to your gpu memory.

            Defaults to 10.
        # file io parameters:

        return_tensor (bool, optional): Whether return the result tensors.

            Defaults to False, will return None.
        output_path (str, optional): output video or gif or image folder.

            Defaults to None, pass export procedure.

        # background frames, priority: image_array > frame_list > origin_frames

        origin_frames (Optional[str], optional): origin background frame path,
            could be `.mp4`, `.gif`(will be sliced into a folder) or an image
            folder.

            Defaults to None.
        frame_list (Optional[List[str]], optional): list of origin background
            frame paths, element in list each should be a image path like
            `*.jpg` or `*.png`.
            Use this when your file names is hard to sort or you only want to
            render a small number frames.

            Defaults to None.
        image_array: (Optional[Union[np.ndarray, torch.Tensor]], optional):
            origin background frame `tensor` or `array`, use this when you
            want your frames in memory as array or tensor.
        overwrite (bool, optional): whether overwriting the existing files.

            Defaults to False.
        mesh_file_path (bool, optional): the directory path to store the `.ply`
            or '.ply' files. Will be named like 'frame_idx_person_idx.ply'.

            Defaults to None.
        read_frames_batch (bool, optional): Whether read frames by batch.
            Set it as True if your video is large in size.

            Defaults to False.

        # visualize keypoints
        plot_pcs (bool, optional): whether plot pointcloud of keypoints on the output video.

            Defaults to False.
        kp3d (Optional[Union[np.ndarray, torch.Tensor]], optional):
            the keypoints of any convention, should pass `mask` if have any
            none-sense points. Shape should be (frame, )

            Defaults to None.
        mask (Optional[Union[np.ndarray, List[int]]], optional):
            Mask of keypoints existence.

            Defaults to None.
        vis_kp_index (bool, optional):
            Whether plot keypoint index number on human mesh.

            Defaults to False.
        # visualize render progress
        verbose (bool, optional):
            Whether print the progress bar for rendering.
    Returns:
        Union[None, torch.Tensor]: return the rendered image tensors or None.
    """
    # initialize the device
    device = torch.device(device) if isinstance(device, str) else device

    RENDER_CONFIGS = mmcv.Config.fromfile(
        os.path.join(
            Path(mmhuman3d.__file__).parents[1],
            'configs/render/smpl.py'))['RENDER_CONFIGS']

    if isinstance(resolution, int):
        resolution = (resolution, resolution)
    elif isinstance(resolution, list):
        resolution = tuple(resolution)
    elif resolution is None:
        if cameras is not None:
            resolution = cameras.resolution[0].tolist()
        else:
            resolution = (1024, 1024)

    verts, poses, betas, transl = _prepare_input_pose(verts, poses, betas,
                                                      transl, device)

    model_type = body_model.name().replace('-', '').lower()

    vertices, joints, num_frames, num_person = _prepare_mesh(
        poses, betas, transl, verts, start, end, body_model)
    end = num_frames if end is None else end
    vertices = vertices.view(num_frames, num_person, -1, 3)
    num_verts = vertices.shape[-2]

    if not plot_pcs:
        joints = None
        if kp3d is not None:
            warnings.warn('`plot_pcs` is False, `kp3d` will be set as None.')
            kp3d = None

    image_array, remove_folder, frames_folder = _prepare_background(
        image_array, frame_list, origin_frames, output_path, start, end,
        img_format, overwrite, num_frames, read_frames_batch, resolution)

    orig_resolution = None
    if image_array is not None:
        orig_resolution = (image_array.shape[1], image_array.shape[2])
        if orig_resolution != resolution:
            image_array = resize_array(image_array, resolution)

    if isinstance(kp3d, np.ndarray):
        kp3d = torch.Tensor(kp3d)

    if kp3d is not None:
        if mask is not None:
            map_index = np.where(np.array(mask) != 0)[0]
            kp3d = kp3d[map_index.tolist()]
        kp3d = kp3d[start:end]
        kp3d = kp3d.view(num_frames, -1, 3)

    if cameras is not None:
        cameras = cameras[start:end]
        cameras.update_resolution_(resolution)
    # prepare render_param_dict
    render_param_dict = copy.deepcopy(RENDER_CONFIGS[render_choice.lower()])
    if model_type == 'smpl':
        render_param_dict.update(num_class=24)
    elif model_type == 'smplx':
        render_param_dict.update(num_class=27)

    if render_choice not in [
            'hq', 'mq', 'lq', 'silhouette', 'part_silhouette', 'depth',
            'pointcloud', 'normal'
    ]:
        raise ValueError('Please choose the right render_choice.')

    # body part colorful visualization should use flat shader to be sharper.
    if texture_image is None:
        if isinstance(palette, str):
            palette = [palette] * num_person
        elif isinstance(palette, (np.ndarray, torch.Tensor, list)):
            if not isinstance(palette, torch.Tensor):
                palette = torch.Tensor(palette)
            palette = palette.view(-1, 3)
            if palette.shape[0] != num_person:
                _times = num_person // palette.shape[0]
                palette = palette.repeat(_times, 1)[:num_person]
                if palette.shape[0] == 1:
                    print(f'Same color for all the {num_person} people')
                else:
                    print('Repeat palette for multi-person.')
        else:
            raise ValueError('Wrong input palette type. '
                             'Palette should be tensor, array or list of strs')
        colors_all = _prepare_colors(palette, render_choice, num_person,
                                     num_verts, model_type)
        colors_all = colors_all.view(-1, num_person * num_verts, 3)
    # verts of ParametricMeshes should be in (N, V, 3)
    vertices = vertices.view(num_frames, -1, 3)
    meshes = ParametricMeshes(
        body_model=body_model,
        verts=vertices,
        N_individual_overdide=num_person,
        model_type=model_type,
        texture_image=texture_image,
        use_nearest=bool(render_choice == 'part_silhouette'),
        vertex_color=colors_all)

    # write .ply or .obj files
    if mesh_file_path is not None:
        mmcv.mkdir_or_exist(mesh_file_path)

        for person_idx in range(meshes.shape[1]):
            mesh_person = meshes[:, person_idx]
            if texture_image is None:
                ply_paths = [
                    f'{mesh_file_path}/frame{frame_idx}_'
                    f'person{person_idx}.ply'
                    for frame_idx in range(num_frames)
                ]
                save_meshes_as_plys(meshes=mesh_person, files=ply_paths)

            else:
                obj_paths = [
                    f'{mesh_file_path}/frame{frame_idx}_'
                    f'person{person_idx}.obj'
                    for frame_idx in range(num_frames)
                ]
                save_meshes_as_objs(meshes=mesh_person, files=obj_paths)

    vertices = meshes.verts_padded().view(num_frames, num_person, -1, 3)

    if num_person > 1:
        vertices = vertices.reshape(num_frames, -1, 3)
    else:
        vertices = vertices.view(num_frames, -1, 3)
    meshes = meshes.update_padded(new_verts_padded=vertices)

    # orig_cam and K are None, use look_at_view
    if cameras is None:
        K, R, T = compute_orbit_cameras(at=(torch.mean(vertices.view(-1, 3),
                                                       0)).detach().cpu(),
                                        orbit_speed=orbit_speed,
                                        batch_size=num_frames,
                                        convention='opencv')
        cameras = build_cameras(
            dict(type='fovperspective',
                 K=K,
                 R=R,
                 T=T,
                 in_ndc=True,
                 device=device,
                 resolution=resolution,
                 Fconvention='pytorch3d'))

    # initialize the renderer.
    renderer = SMPLRenderer(resolution=resolution,
                            device=device,
                            output_path=output_path,
                            return_tensor=return_tensor,
                            alpha=alpha,
                            read_img_format=img_format,
                            render_choice=render_choice,
                            frames_folder=frames_folder,
                            plot_pcs=plot_pcs,
                            vis_kp_index=vis_kp_index,
                            **render_param_dict)

    if image_array is not None:
        image_array = align_input_to_padded(image_array,
                                            ndim=4,
                                            batch_size=num_frames,
                                            padding_mode='ones')
    # prepare the render data.
    render_data = dict(
        images=image_array,
        meshes=meshes,
        cameras=cameras,
        joints=joints,
        joints_gt=kp3d,
    )

    results = render_runner.render(renderer=renderer,
                                   device=device,
                                   batch_size=batch_size,
                                   output_path=output_path,
                                   return_tensor=return_tensor,
                                   no_grad=no_grad,
                                   verbose=verbose,
                                   **render_data)
    if remove_folder:
        if Path(frames_folder).is_dir():
            shutil.rmtree(frames_folder)

    if return_tensor:
        return results
    else:
        return None
