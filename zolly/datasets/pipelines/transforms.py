import random

import cv2
import mmcv
import numpy as np

from zolly.models.body_models.mappings import get_flip_pairs, convert_kps
from mmhuman3d.data.datasets.pipelines.compose import Compose
from mmhuman3d.data.datasets.pipelines.transforms import (
    get_affine_transform, _flip_axis_angle, _flip_hand_pose, _flip_keypoints,
    _flip_smpl_pose, _flip_smplx_pose, _rotate_joints_3d, _rotate_smpl_pose,
    affine_transform)
from zolly.utils.bbox_utils import kp2d_to_bbox


class ResizeSample:
    """Affine transform the image to get input image.

    Affine transform the 2D keypoints, 3D kepoints. Required keys: 'img',
    'pose', 'img_shape', 'rotation' and 'center'. Modifies key: 'img',
    ''keypoints2d', 'keypoints3d', 'pose'.
    """

    def __init__(self,
                 scale=1.0,
                 dst_hight=None,
                 dst_size=None,
                 img_fields=['img'],
                 backend='cv2',
                 interpolation=dict()):
        self.scale = scale
        self.dst_hight = dst_hight
        self.img_fields = img_fields
        self.interpolation = interpolation
        self.backend = backend
        self.dst_size = dst_size
        for v in interpolation.values():
            assert v in ('nearest', 'bilinear', 'bicubic', 'area', 'lanczos')
        if backend not in ['cv2', 'pillow']:
            raise ValueError(f'backend: {backend} is not supported for '
                             'resize. Supported backends are "cv2", "pillow"')

    def __call__(self, results):

        if self.dst_hight is not None:
            scale_factor = results['ori_shape'][0] / self.dst_hight
            scale_factor = (scale_factor, scale_factor)
            dst_size = (self.dst_hight, results['ori_shape'][1] / scale_factor)
        elif self.dst_size is not None:
            scale_factor = (results['ori_shape'][1] / self.dst_size[1],
                            results['ori_shape'][0] / self.dst_size[0])
            dst_size = self.dst_size
        else:
            scale_factor = (self.scale, self.scale)
            dst_size = (results['ori_shape'][1] / self.scale,
                        results['ori_shape'][0] / self.scale)

        if scale_factor != 1.0:
            if 'center' in results:
                c = results['center']
                c = c / np.array(scale_factor).reshape(-1)
                results['center'] = c
            if 'scale' in results:
                s = results['scale']
                s = s / np.array(scale_factor).reshape(-1)
                results['scale'] = s

            for part_name in ['lhand', 'rhand', 'face']:
                if f'{part_name}_center' in results:
                    c = results[f'{part_name}_center']
                    c = c / np.array(scale_factor).reshape(-1)
                    results[f'{part_name}_center'] = c
                if f'{part_name}_scale' in results:
                    s = results[f'{part_name}_scale']
                    s = s / np.array(scale_factor).reshape(-1)
                    results[f'{part_name}_scale'] = s

            for key in results.get('img_fields', self.img_fields):
                img = results[key]
                dst_h, dst_w = dst_size

                interpolation = self.interpolation.get(key, 'bilinear')
                img = mmcv.imresize(img, (dst_w, dst_h),
                                    interpolation=interpolation,
                                    backend=self.backend)
                results[key] = img

            if 'keypoints2d' in results:
                keypoints2d = results['keypoints2d'].copy()
                num_keypoints = len(keypoints2d)
                for i in range(num_keypoints):
                    if keypoints2d[i][2] > 0.0:
                        keypoints2d[i][:2] = keypoints2d[i][:2] / np.array(
                            scale_factor).reshape(1, 2)

                results['keypoints2d'] = keypoints2d

            return results
        else:
            return results


class GetRandomScaleRotation:
    """Data augmentation with random scaling & rotating.

    Required key: 'scale'. Modifies key: 'scale' and 'rotation'.
    Args:
        rot_factor (int): Rotating to ``[-2*rot_factor, 2*rot_factor]``.
        scale_factor (float): Scaling to ``[1-scale_factor, 1+scale_factor]``.
        rot_prob (float): Probability of random rotation.
    """

    def __init__(self, rot_factor=30, scale_factor=0.25, rot_prob=0.6):
        self.rot_factor = rot_factor
        self.scale_factor = scale_factor
        self.rot_prob = rot_prob

    def __call__(self, results):
        """Perform data augmentation with random scaling & rotating."""

        sf = self.scale_factor
        rf = self.rot_factor

        s_factor = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

        s = results['scale']
        s = s * s_factor
        results['scale'] = s

        for part_name in ['lhand', 'rhand', 'face']:
            if f'{part_name}_scale' in results:
                s = results[f'{part_name}_scale']
                s = s * s_factor
                results[f'{part_name}_scale'] = s

        r_factor = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
        r = r_factor if np.random.rand() <= self.rot_prob else 0.0

        results['rotation'] = r

        return results


class RandomHorizontalFlip(object):
    """Flip the image randomly.

    Flip the image randomly based on flip probaility.

    Args:
        flip_prob (float): probability of the image being flipped. Default: 0.5
    """

    def __init__(self, flip_prob=0.5, convention='coco', img_fields=['img']):
        assert 0 <= flip_prob <= 1
        self.flip_prob = flip_prob
        self.flip_pairs = get_flip_pairs(convention)
        self.img_fields = img_fields

    def __call__(self, results):
        """Call function to flip image and annotations.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip' key is added into
                result dict.
        """
        if np.random.rand() > self.flip_prob:
            results['is_flipped'] = np.array([0])
            return results

        results['is_flipped'] = np.array([1])

        if results['has_K']:
            W = results['ori_shape'][1]
            K = results['K']
            cx = K[0, 2]
            K[0, 2] = W - cx
            results['K'] = K

        # flip image
        for key in results.get('img_fields', self.img_fields):
            results[key] = mmcv.imflip(results[key], direction='horizontal')
            if (key == 'iuv_img'):
                results[key][...,
                             1] = (1 -
                                   results[key][..., 1]) * results[key][..., 0]
            elif key == 'mask':
                results[key] = results[key].copy()
        if 'smpl_transl' in results:
            transl = results['smpl_transl'].copy()
            transl = np.concatenate([transl[..., 0:1] * -1, transl[..., 1:]],
                                    -1)
            results['smpl_transl'] = transl
        if 'smplx_transl' in results:
            transl = results['smplx_transl'].copy()
            transl = np.concatenate([transl[..., 0:1] * -1, transl[..., 1:]],
                                    -1)
            results['smplx_transl'] = transl

        if 'smpl_origin_orient' in results:
            smpl_origin_orient = results['smpl_origin_orient'].copy()
            results['smpl_origin_orient'] = smpl_origin_orient * np.array(
                [1, -1, -1], dtype=np.float32)

        # flip keypoints2d
        if 'keypoints2d' in results:
            assert self.flip_pairs is not None
            width = results['img'][:, ::-1, :].shape[1]
            keypoints2d = results['keypoints2d'].copy()
            keypoints2d = _flip_keypoints(keypoints2d, self.flip_pairs, width)
            results['keypoints2d'] = keypoints2d

        # flip bbox center
        if 'center' in results:
            center = results['center']
            center[0] = width - 1 - center[0]
            results['center'] = center

        for part_name in ['lhand', 'rhand', 'face']:
            if f'{part_name}_center' in results:
                center = results[f'{part_name}_center']
                center[0] = width - 1 - center[0]
                results[f'{part_name}_center'] = center
        if 'lhand_center' in results and 'rhand_center' in results:
            lhand_center = results['lhand_center'].copy()
            rhand_center = results['rhand_center'].copy()
            lhand_scale = results['lhand_scale'].copy()
            rhand_scale = results['rhand_scale'].copy()
            results['lhand_center'] = rhand_center
            results['rhand_center'] = lhand_center
            results['lhand_scale'] = rhand_scale
            results['rhand_scale'] = lhand_scale

        # flip keypoints3d
        if 'keypoints3d' in results:
            assert self.flip_pairs is not None
            keypoints3d = results['keypoints3d'].copy()
            keypoints3d = _flip_keypoints(keypoints3d, self.flip_pairs)
            results['keypoints3d'] = keypoints3d

        # flip smpl
        if 'smpl_body_pose' in results:
            global_orient = results['smpl_global_orient'].copy()
            body_pose = results['smpl_body_pose'].copy().reshape((-1))
            smpl_pose = np.concatenate((global_orient, body_pose), axis=-1)
            smpl_pose_flipped = _flip_smpl_pose(smpl_pose)
            global_orient = smpl_pose_flipped[:3]
            body_pose = smpl_pose_flipped[3:]
            results['smpl_global_orient'] = global_orient
            results['smpl_body_pose'] = body_pose.reshape((-1, 3))

        if 'smplx_body_pose' in results:

            body_pose = results['smplx_body_pose'].copy().reshape((-1))
            body_pose_flipped = _flip_smplx_pose(body_pose)
            results['smplx_body_pose'] = body_pose_flipped

        if 'smplx_global_orient' in results:
            global_orient = results['smplx_global_orient'].copy().reshape((-1))
            global_orient_flipped = _flip_axis_angle(global_orient)
            results['smplx_global_orient'] = global_orient_flipped

        if 'smplx_jaw_pose' in results:
            jaw_pose = results['smplx_jaw_pose'].copy().reshape((-1))
            jaw_pose_flipped = _flip_axis_angle(jaw_pose)
            results['smplx_jaw_pose'] = jaw_pose_flipped

        if 'smplx_right_hand_pose' in results:

            right_hand_pose = results['smplx_right_hand_pose'].copy()
            left_hand_pose = results['smplx_left_hand_pose'].copy()
            results['smplx_right_hand_pose'], results[
                'smplx_left_hand_pose'] = _flip_hand_pose(
                    right_hand_pose, left_hand_pose)

        # Expressions are not symmetric. Remove them when flipped.
        if 'smplx_expression' in results:
            results['smplx_expression'] = np.zeros(
                (results['smplx_expression'].shape[0]), dtype=np.float32)
            results['has_smplx_expression'] = 0

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_prob={self.flip_prob})'


class CenterCrop(object):
    r"""Center crop the image.

    Args:
        crop_size (int | tuple): Expected size after cropping with the format
            of (h, w).
        efficientnet_style (bool): Whether to use efficientnet style center
            crop. Defaults to False.
        crop_padding (int): The crop padding parameter in efficientnet style
            center crop. Only valid if efficientnet style is True. Defaults to
            32.
        interpolation (str): Interpolation method, accepted values are
            'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Only valid if
             efficientnet style is True. Defaults to 'bilinear'.
        backend (str): The image resize backend type, accpeted values are
            `cv2` and `pillow`. Only valid if efficientnet style is True.
            Defaults to `cv2`.


    Notes:
        If the image is smaller than the crop size, return the original image.
        If efficientnet_style is set to False, the pipeline would be a simple
        center crop using the crop_size.
        If efficientnet_style is set to True, the pipeline will be to first to
        perform the center crop with the crop_size_ as:

        .. math::
        crop\_size\_ = crop\_size / (crop\_size + crop\_padding) * short\_edge

        And then the pipeline resizes the img to the input crop size.
    """

    def __init__(self,
                 crop_size,
                 efficientnet_style=False,
                 crop_padding=32,
                 interpolation=dict(),
                 backend='cv2',
                 img_fields=['img']):
        if efficientnet_style:
            assert isinstance(crop_size, int)
            assert crop_padding >= 0
            for v in interpolation.values():
                assert v in ('nearest', 'bilinear', 'bicubic', 'area',
                             'lanczos')
            if backend not in ['cv2', 'pillow']:
                raise ValueError(
                    f'backend: {backend} is not supported for '
                    'resize. Supported backends are "cv2", "pillow"')
        else:
            assert isinstance(crop_size, int) or (isinstance(crop_size, tuple)
                                                  and len(crop_size) == 2)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.efficientnet_style = efficientnet_style
        self.crop_padding = crop_padding
        self.interpolation = interpolation
        self.backend = backend
        self.img_fields = img_fields

    def __call__(self, results):
        crop_height, crop_width = self.crop_size[0], self.crop_size[1]
        for key in self.img_fields:
            img = results[key]
            # img.shape has length 2 for grayscale, length 3 for color
            img_height, img_width = img.shape[:2]

            # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/preprocessing.py#L118 # noqa
            if self.efficientnet_style:
                img_short = min(img_height, img_width)
                crop_height = crop_height / (crop_height +
                                             self.crop_padding) * img_short
                crop_width = crop_width / (crop_width +
                                           self.crop_padding) * img_short

            y1 = max(0, int(round((img_height - crop_height) / 2.)))
            x1 = max(0, int(round((img_width - crop_width) / 2.)))
            y2 = min(img_height, y1 + crop_height) - 1
            x2 = min(img_width, x1 + crop_width) - 1

            # crop the image
            img = mmcv.imcrop(img, bboxes=np.array([x1, y1, x2, y2]))

            if self.efficientnet_style:
                interpolation = self.interpolation.get(key, 'bilinear')
                img = mmcv.imresize(img,
                                    tuple(self.crop_size[::-1]),
                                    interpolation=interpolation,
                                    backend=self.backend)
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(crop_size={self.crop_size}'
        repr_str += f', efficientnet_style={self.efficientnet_style}'
        repr_str += f', crop_padding={self.crop_padding}'
        repr_str += f', interpolation={self.interpolation}'
        repr_str += f', backend={self.backend})'
        return repr_str


class Normalize(object):
    """Normalize the image.

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True, img_fields=['img']):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.img_fields = img_fields

    def __call__(self, results):
        for key in results.get('img_fields', self.img_fields):
            results[key] = mmcv.imnormalize(results[key], self.mean, self.std,
                                            self.to_rgb)
        results['img_norm_cfg'] = dict(mean=self.mean,
                                       std=self.std,
                                       to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={list(self.mean)}, '
        repr_str += f'std={list(self.std)}, '
        repr_str += f'to_rgb={self.to_rgb})'
        return repr_str


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float): How much to jitter brightness.
            brightness_factor is chosen uniformly from
            [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast.
            contrast_factor is chosen uniformly from
            [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation.
            saturation_factor is chosen uniformly from
            [max(0, 1 - saturation), 1 + saturation].
    """

    def __init__(self, brightness, contrast, saturation):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, results):
        brightness_factor = random.uniform(0, self.brightness)
        contrast_factor = random.uniform(0, self.contrast)
        saturation_factor = random.uniform(0, self.saturation)
        color_jitter_transforms = [
            dict(type='Brightness',
                 magnitude=brightness_factor,
                 prob=1.,
                 random_negative_prob=0.5),
            dict(type='Contrast',
                 magnitude=contrast_factor,
                 prob=1.,
                 random_negative_prob=0.5),
            dict(type='ColorTransform',
                 magnitude=saturation_factor,
                 prob=1.,
                 random_negative_prob=0.5)
        ]
        random.shuffle(color_jitter_transforms)
        transform = Compose(color_jitter_transforms)
        return transform(results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(brightness={self.brightness}, '
        repr_str += f'contrast={self.contrast}, '
        repr_str += f'saturation={self.saturation})'
        return repr_str


class Lighting(object):
    """Adjust images lighting using AlexNet-style PCA jitter.

    Args:
        eigval (list): the eigenvalue of the convariance matrix of pixel
            values, respectively.
        eigvec (list[list]): the eigenvector of the convariance matrix of pixel
            values, respectively.
        alphastd (float): The standard deviation for distribution of alpha.
            Defaults to 0.1
        to_rgb (bool): Whether to convert img to rgb.
    """

    def __init__(self,
                 eigval,
                 eigvec,
                 alphastd=0.1,
                 to_rgb=True,
                 img_fields=['img']):
        assert isinstance(eigval, list), \
            f'eigval must be of type list, got {type(eigval)} instead.'
        assert isinstance(eigvec, list), \
            f'eigvec must be of type list, got {type(eigvec)} instead.'
        for vec in eigvec:
            assert isinstance(vec, list) and len(vec) == len(eigvec[0]), \
                'eigvec must contains lists with equal length.'
        self.eigval = np.array(eigval)
        self.eigvec = np.array(eigvec)
        self.alphastd = alphastd
        self.to_rgb = to_rgb
        self.img_fields = img_fields

    def __call__(self, results):
        for key in results.get('img_fields', self.img_fields):
            img = results[key]
            results[key] = mmcv.adjust_lighting(img,
                                                self.eigval,
                                                self.eigvec,
                                                alphastd=self.alphastd,
                                                to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(eigval={self.eigval.tolist()}, '
        repr_str += f'eigvec={self.eigvec.tolist()}, '
        repr_str += f'alphastd={self.alphastd}, '
        repr_str += f'to_rgb={self.to_rgb})'
        return repr_str


class RandomChannelNoise:
    """Data augmentation with random channel noise.

    Required keys: 'img'
    Modifies key: 'img'
    Args:
        noise_factor (float): Multiply each channel with
         a factor between``[1-scale_factor, 1+scale_factor]``
    """

    def __init__(self, noise_factor=0.4, img_fields=['img']):
        self.noise_factor = noise_factor
        self.img_fields = img_fields

    def __call__(self, results):
        """Perform data augmentation with random channel noise."""

        # Each channel is multiplied with a number
        # in the area [1-self.noise_factor, 1+self.noise_factor]
        pn = np.random.uniform(1 - self.noise_factor, 1 + self.noise_factor,
                               (1, 3))

        for key in results.get('img_fields', self.img_fields):
            img = results[key]
            results[key] = cv2.multiply(img, pn)

        return results


class MeshAffine:
    """Affine transform the image to get input image.

    Affine transform the 2D keypoints, 3D kepoints. Required keys: 'img',
    'pose', 'img_shape', 'rotation' and 'center'. Modifies key: 'img',
    ''keypoints2d', 'keypoints3d', 'pose'.
    """

    def __init__(self,
                 img_res=dict(img=224),
                 img_fields=['img'],
                 require_origin_kp2d=True,
                 convention_src='smplx',
                 convention_dst=dict(),
                 erase_background=False):
        self.image_size = img_res
        self.img_fields = img_fields
        self.require_origin_kp2d = require_origin_kp2d
        self.convention_src = convention_src
        self.convention_dst = convention_dst
        self.erase_background = erase_background

    def get_res(self, key):
        if isinstance(self.image_size, (int, float)):
            res = (int(self.image_size), int(self.image_size))
        elif isinstance(self.image_size, (tuple, list)):
            res = self.image_size
        elif isinstance(self.image_size, dict):
            res = self.image_size.get(key, self.image_size.get('img'))
            res = (int(res), int(res)) if isinstance(res,
                                                     (float, int)) else res
        return res

    def __call__(self, results):
        c = results['center']
        s = results['scale']
        r = results['rotation']
        trans = get_affine_transform(c, s, r, self.get_res('img'))
        trans_dict = dict(body=trans)
        if 'lhand_img' in self.img_fields:
            lhand_c = results.get('lhand_center', None)
            lhand_s = results.get('lhand_scale', None)
            lhand_r = results.get('lhand_rotation', r)
            lhand_trans = get_affine_transform(lhand_c, lhand_s, lhand_r,
                                               self.get_res('lhand_img'))
            trans_dict.update(lhand=lhand_trans)
        if 'rhand_img' in self.img_fields:
            rhand_c = results.get('rhand_center', None)
            rhand_s = results.get('rhand_scale', None)
            rhand_r = results.get('rhand_rotation', r)
            rhand_trans = get_affine_transform(rhand_c, rhand_s, rhand_r,
                                               self.get_res('rhand_img'))
            trans_dict.update(rhand=rhand_trans)
        if 'face_img' in self.img_fields:
            face_c = results.get('face_center', None)
            face_s = results.get('face_scale', None)
            face_r = results.get('face_rotation', r)
            face_trans = get_affine_transform(face_c, face_s, face_r,
                                              self.get_res('face_img'))
            trans_dict.update(face=face_trans)
        results['crop_transform'] = trans
        inv_trans = get_affine_transform(c,
                                         s,
                                         0.,
                                         self.get_res('img'),
                                         inv=True)
        crop_trans = get_affine_transform(c, s, 0., self.get_res('img'))

        results['trans'] = trans
        results['crop_trans_worot'] = crop_trans
        results['inv_trans'] = inv_trans

        ori_img = results['img'].copy()

        if self.erase_background:
            xyxy = kp2d_to_bbox(results['keypoints2d'], scale_factor=1.1)
            new_ori_img = np.random.randint(0, 255,
                                            ori_img.shape).astype(np.uint8)
            left, top, right, bottom = xyxy.reshape(4).tolist()
            left, top, right, bottom = int(left), int(top), int(right), int(
                bottom)
            new_ori_img[top:bottom, left:right] = ori_img[top:bottom,
                                                          left:right]
            ori_img = new_ori_img
        for key in results.get('img_fields', self.img_fields):
            if key == 'ori_img':
                results['ori_img'] = ori_img
            if key == 'img':
                res = self.get_res(key)
                results[key] = cv2.warpAffine(ori_img,
                                              trans, (res[1], res[0]),
                                              flags=cv2.INTER_LINEAR)
                # if default_res != res:
                #     results[key] = cv2.resize(results[key], (res[1], res[0]),
                #                               cv2.INTER_LINEAR)
            elif key == 'lhand_img':
                res = self.get_res(key)
                results[key] = cv2.warpAffine(ori_img,
                                              lhand_trans, (res[1], res[0]),
                                              flags=cv2.INTER_LINEAR)
            elif key == 'rhand_img':
                res = self.get_res(key)
                results[key] = cv2.warpAffine(ori_img,
                                              rhand_trans, (res[1], res[0]),
                                              flags=cv2.INTER_LINEAR)
            elif key == 'face_img':
                res = self.get_res(key)
                results[key] = cv2.warpAffine(ori_img,
                                              face_trans, (res[1], res[0]),
                                              flags=cv2.INTER_LINEAR)
            elif key == 'full_img':
                res = self.get_res(key)
                h, w = results['ori_shape']
                full_trans = get_affine_transform(np.array([w / 2, h / 2]),
                                                  np.array([h, h]), 0, res)
                results[key] = cv2.warpAffine(ori_img,
                                              full_trans, (res[1], res[0]),
                                              flags=cv2.INTER_LINEAR)

        if 'keypoints2d' in results:
            origin_keypoints2d = results['keypoints2d'].copy()
            if self.require_origin_kp2d:
                results['ori_keypoints2d'] = origin_keypoints2d

            keypoints2d_full = results['keypoints2d'].copy()
            num_keypoints = len(keypoints2d_full)
            for i in range(num_keypoints):
                if keypoints2d_full[i][2] > 0.0:
                    keypoints2d_full[i][:2] = \
                        affine_transform(keypoints2d_full[i][:2], trans)
            results['keypoints2d'] = keypoints2d_full

            for part_name in self.convention_dst:
                keypoints2d_part, _ = convert_kps(
                    origin_keypoints2d[None], self.convention_src,
                    self.convention_dst[part_name])
                keypoints2d_part = keypoints2d_part[0]
                num_keypoints = len(keypoints2d_part)

                for i in range(num_keypoints):
                    if keypoints2d_part[i][2] > 0.0:
                        keypoints2d_part[i][:2] = \
                            affine_transform(keypoints2d_part[i][:2], trans_dict[part_name])
                results[f'keypoints2d_{part_name}'] = keypoints2d_part

        if 'keypoints3d' in results:
            keypoints3d = results['keypoints3d'].copy()
            keypoints3d[:, :3] = _rotate_joints_3d(keypoints3d[:, :3], r)
            results['keypoints3d'] = keypoints3d

            for part_name in self.convention_dst:
                keypoints3d_part, _ = convert_kps(
                    keypoints3d[None], self.convention_src,
                    self.convention_dst[part_name])
                results[f'keypoints3d_{part_name}'] = keypoints3d_part[0]

        if 'smpl_body_pose' in results:
            global_orient = results['smpl_global_orient'].copy()
            body_pose = results['smpl_body_pose'].copy().reshape((-1))
            pose = np.concatenate((global_orient, body_pose), axis=-1)
            pose = _rotate_smpl_pose(pose, r)
            results['smpl_global_orient'] = pose[:3]
            results['smpl_body_pose'] = pose[3:].reshape((-1, 3))

        if 'smplx_global_orient' in results:
            global_orient = results['smplx_global_orient'].copy()
            global_orient = _rotate_smpl_pose(global_orient, r)
            results['smplx_global_orient'] = global_orient

        return results


class Rotation:
    """Rotate the image with the given rotation.

    Rotate the 2D keypoints, 3D kepoints, poses. Required keys: 'img',
    'pose', 'rotation' and 'center'. Modifies key: 'img',
    ''keypoints2d', 'keypoints3d', 'pose'.

    To avoid conflicts with MeshAffine, rotation will be set to 0.0
    after rotate the image.
    The rotation value will be stored to 'ori_rotation'.
    """

    def __init__(self):
        pass

    def __call__(self, results):
        r = results['rotation']
        if r == 0.0:
            return results
        img = results['img']

        # img before affine
        (h, w) = img.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), r, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        # perform the actual rotation and return the image
        img = cv2.warpAffine(img, M, (nW, nH))
        results['img'] = img

        for part_name in ['lhand', 'rhand', 'face']:
            if f'{part_name}_img' in results:
                img_part = results[f'{part_name}_img']
                img_part = cv2.warpAffine(img_part, M, (nW, nH))
                results[f'{part_name}_img'] = img_part

        c = results['center']
        c = np.dot(M[:2, :2], c) + M[:2, 2]
        results['center'] = c

        for part_name in ['lhand', 'rhand', 'face']:
            c = results[f'{part_name}_center']
            c = np.dot(M[:2, :2], c) + M[:2, 2]
            results[f'{part_name}_center'] = c

        if 'keypoints2d' in results:
            keypoints2d = results['keypoints2d'].copy()
            keypoints2d[:, :2] = (np.dot(keypoints2d[:, :2], M[:2, :2].T) +
                                  M[:2, 2] + 1).astype(np.int)
            results['keypoints2d'] = keypoints2d
            for part_name in ['lhand', 'rhand', 'face']:
                if f'keypoints2d_{part_name}' in results:
                    keypoints2d_part = results[
                        f'keypoints2d_{part_name}'].copy()
                    keypoints2d_part[:, :2] = (
                        np.dot(keypoints2d_part[:, :2], M[:2, :2].T) +
                        M[:2, 2] + 1).astype(np.int)
                    results[f'keypoints2d_{part_name}'] = keypoints2d_part

        if 'keypoints3d' in results:
            keypoints3d = results['keypoints3d'].copy()
            keypoints3d[:, :3] = _rotate_joints_3d(keypoints3d[:, :3], r)
            results['keypoints3d'] = keypoints3d

            for part_name in ['lhand', 'rhand', 'face']:
                if f'keypoints3d_{part_name}' in results:
                    keypoints3d_part = results[
                        f'keypoints3d_{part_name}'].copy()
                    keypoints3d_part[:, :3] = _rotate_joints_3d(
                        keypoints3d_part[:, :3], r)
                    results[f'keypoints3d_{part_name}'] = keypoints3d_part

        if 'smpl_body_pose' in results:
            global_orient = results['smpl_global_orient'].copy()
            body_pose = results['smpl_body_pose'].copy().reshape((-1))
            pose = np.concatenate((global_orient, body_pose), axis=-1)
            pose = _rotate_smpl_pose(pose, r)
            results['smpl_global_orient'] = pose[:3]
            results['smpl_body_pose'] = pose[3:].reshape((-1, 3))

        if 'smplx_global_orient' in results:
            global_orient = results['smplx_global_orient'].copy()
            global_orient = _rotate_smpl_pose(global_orient, r)
            results['smplx_global_orient'] = global_orient

        results['rotation'] = 0.0
        results['ori_rotation'] = r
        return results


class GetBboxInfo(object):
    """Get bbox for cliff.
    Required key: 'scale'. Modifies key: 'scale' and 'rotation'.
    Args:
        rot_factor (int): Rotating to ``[-2*rot_factor, 2*rot_factor]``.
        scale_factor (float): Scaling to ``[1-scale_factor, 1+scale_factor]``.
        rot_prob (float): Probability of random rotation.
    """

    def estimate_focal_length(self, img_h, img_w):
        return (img_w * img_w + img_h * img_h)**0.5  # fov: 55 degree

    def __call__(self, results):
        """(1) Get focal length from original image (2) get bbox_info from c
        and s."""
        img_h, img_w = results['ori_shape']
        if not results['has_focal_length']:
            focal_length_pixel = self.estimate_focal_length(img_h, img_w)
            results['ori_focal_length'] = focal_length_pixel
        else:
            focal_length_pixel = results['ori_focal_length']

        results['img_h'] = img_h
        results['img_w'] = img_w

        cx, cy = results['center']
        s = results['scale'][0]

        bbox_info = np.stack([cx - img_w / 2., cy - img_h / 2., s])
        bbox_info[:2] = bbox_info[:2] / focal_length_pixel * 2.8  # [-1, 1]
        bbox_info[2] = (bbox_info[2] - 0.24 * focal_length_pixel) / (
            0.06 * focal_length_pixel)  # [-1, 1]

        results['bbox_info'] = np.float32(bbox_info)

        return results


class GenHeatmap(object):

    def __init__(
            self,
            sigma,
            down_scale=4,
            img_size=256,
            model_type='smpl',
            img_name_src=dict(),
    ):
        self.sigma = sigma
        self.gaussian = self._get_gaussian()
        self.down_scale = down_scale
        self.model_type = model_type
        self.img_name_src = img_name_src
        self.img_size = img_size if isinstance(img_size,
                                               (list,
                                                tuple)) else (int(img_size),
                                                              int(img_size))

    def _get_gaussian(self):
        sigma = self.sigma
        tmp_size = sigma * 3
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    def __call__(self, results):
        if self.model_type == 'smpl':
            transl = results.get('smpl_transl')
        else:
            transl = results.get('smplx_transl', None)

        for part_name, img_name in self.img_name_src.items():

            keypoints2d_for_heatmap = results[f'keypoints2d_{part_name}']
            keypoints3d_part = results[f'keypoints3d_{part_name}']

            mask = keypoints2d_for_heatmap[..., 2]

            num_joints = keypoints2d_for_heatmap.shape[0]
            h, w = results[img_name].shape[:2]
            target = np.zeros(
                (num_joints, h // self.down_scale, w // self.down_scale),
                dtype=np.float32)

            distortion_scale = transl[2] / (transl[2] + keypoints3d_part[:, 2])

            target_weight = np.ones(
                (num_joints, 1), dtype=np.float32) * mask.reshape(-1, 1)
            if (keypoints2d_for_heatmap is not None):
                target_weight[:, 0] = keypoints2d_for_heatmap[:, 2] > 0.5
                sigma = self.sigma
                tmp_size = sigma * 3
                for joint_id in range(num_joints):
                    d_scale = distortion_scale[joint_id]
                    cx = int(keypoints2d_for_heatmap[joint_id][0] //
                             self.down_scale)
                    cy = int(keypoints2d_for_heatmap[joint_id][1] //
                             self.down_scale)
                    left = cx - int(tmp_size * d_scale)
                    up = cy - int(tmp_size * d_scale)

                    right = cx + int(tmp_size * d_scale) + 1
                    bottom = cy + int(tmp_size * d_scale) + 1

                    if left >= w/self.down_scale or up >= h/self.down_scale \
                            or right < 0 or bottom < 0:
                        # If not, just return the image as is
                        target_weight[joint_id] = 0
                        continue

                    # Usable gaussian range
                    gx = max(0, -left), min(right, w // self.down_scale) - left
                    gy = max(0, -up), min(bottom, h // self.down_scale) - up
                    # Image range
                    img_x = max(0, left), min(right, w // self.down_scale)
                    img_y = max(0, up), min(bottom, h // self.down_scale)

                    v = target_weight[joint_id]
                    if v:
                        gaussian = cv2.resize(
                            self.gaussian, (int(tmp_size * d_scale * 2 + 1),
                                            int(tmp_size * d_scale * 2 + 1)),
                            cv2.INTER_CUBIC)
                        target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                            gaussian[gy[0]:gy[1], gx[0]:gx[1]]

            results[f'{part_name}_heatmap'] = target
            results[f'weight_heatmap_{part_name}'] = target_weight
        return results
