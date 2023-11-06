import mmdet
import mmpose
import os
from tqdm import trange
import math
from pathlib import Path
import numpy as np

root = str(Path(mmdet.__path__[0]).parent.absolute())


def get_config(object_type):
    assert object_type in ['hand', 'body', 'face']
    if object_type == 'hand':
        det_config = f"{str(Path(mmpose.__path__[0]).parent.absolute())}/demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py"
        det_checkpoint = "https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth"
    elif object_type == 'body':
        det_config = "configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"
        det_checkpoint = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

    return det_config, det_checkpoint


def inference_bbox(imgs_or_paths,
                   batch_size=5,
                   device='cpu',
                   object_type='body',
                   single_object=True):
    from mmdet.apis import inference_detector, init_detector
    from mmpose.apis import process_mmdet_results
    if object_type == 'whole_body':
        object_type = 'body'
    det_config, det_checkpoint = get_config(object_type)
    det_config = os.path.join(root, det_config)

    det_model = init_detector(det_config, det_checkpoint, device=device)
    # build the pose model from a config file and a checkpoint file
    mmdet_results = []
    # test a single image, the resulting box is (x1, y1, x2, y2)
    for i in trange(math.ceil(len(imgs_or_paths) / batch_size)):
        current_imgs = imgs_or_paths[i *
                                     batch_size:min(len(imgs_or_paths), (i +
                                                                         1) *
                                                    batch_size)]
        mmdet_results += inference_detector(det_model, list(current_imgs))

    bboxes = []
    for mmdet_res in mmdet_results:
        if not single_object:
            bboxes.append(process_mmdet_results(mmdet_res, 1))
        else:
            bboxes.append(process_mmdet_results(mmdet_res, 1)[0]['bbox'][None])
    if single_object:
        bboxes = np.concatenate(bboxes)
    return bboxes
