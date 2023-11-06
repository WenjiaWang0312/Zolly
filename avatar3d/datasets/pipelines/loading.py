import mmcv
import numpy as np
import os.path as osp
from mmhuman3d.data.datasets.pipelines import LoadImageFromFile


class LoadUVDfromFile(LoadImageFromFile):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`) and "img_norm_cfg" (means=0 and stds=1).
    Both "img_shape" and "ori_shape" use (height, width) convention.

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes()`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=True,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'], results['image_path'])
            if results['has_uvd']:
                uvd_filename = osp.join(results['img_prefix'],
                                        results['uvd_path'])
        else:
            filename = results['image_path']
            if results['has_uvd']:
                uvd_filename = results['uvd_path']
            else:
                uvd_filename = 'none'

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)

        if results['has_uvd']:
            uvd_img_bytes = self.file_client.get(uvd_filename)
            uvd_img = mmcv.imfrombytes(uvd_img_bytes, flag=self.color_type)
        else:
            uvd_img = np.zeros_like(img)

        if self.to_float32:
            img = img.astype(np.float32)
            uvd_img = uvd_img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['image_path']
        results['uvd_filename'] = uvd_filename
        results['img'] = img

        ori_shape = img.shape[:2]

        results['ori_shape'] = ori_shape

        uv = uvd_img[..., :2] / 255.0

        d_img = uvd_img[..., 2:3] / 255.0 * results['distortion_max']
        mask = (d_img > 0.2) * 1.0
        iuv_img = np.concatenate([mask, uv], -1)
        results['iuv_img'] = iuv_img
        results['d_img'] = d_img

        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(mean=np.zeros(num_channels,
                                                     dtype=np.float32),
                                       std=np.ones(num_channels,
                                                   dtype=np.float32),
                                       to_rgb=False)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


class LoadSegfromFile(LoadImageFromFile):

    def __init__(self,
                 to_float32=True,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'], results['image_path'])
            if results['has_mask']:
                mask_filename = osp.join(results['img_prefix'],
                                         results['mask_path'])
        else:
            filename = results['image_path']
            if results['has_mask']:
                mask_filename = results['mask_path']
            else:
                mask_filename = 'none'

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)

        if results['has_mask']:
            mask_img_bytes = self.file_client.get(mask_filename)
            mask_img = mmcv.imfrombytes(mask_img_bytes, flag=self.color_type)
            mask_img = (mask_img[..., 0:1] > 0) * 1.0
        else:
            mask_img = np.zeros_like(img)[..., 0:1]

        if self.to_float32:
            img = img.astype(np.float32)
            mask_img = mask_img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['image_path']
        results['ori_mask_filename'] = results['mask_path']
        results['img'] = img
        results['mask'] = mask_img
        results['ori_shape'] = img.shape[:2]
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(mean=np.zeros(num_channels,
                                                     dtype=np.float32),
                                       std=np.ones(num_channels,
                                                   dtype=np.float32),
                                       to_rgb=False)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str
