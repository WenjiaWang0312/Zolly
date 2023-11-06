import mmseg
from pathlib import Path
import glob
import torch
import math
from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor

root = str(Path(mmseg.__path__[0]).parent.absolute())


def inference_human_mask(
        im_root='',
        config='configs/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes.py',
        checkpoint='deeplabv3_r50-d8_512x1024_40k_cityscapes_20200605_022449-acadc2f8.pth',
        device=torch.device('cuda'),
        batch_size=1,
):

    # build the model from a config file and a checkpoint file
    model = init_segmentor(f'{root}/{config}',
                           f'{root}/{checkpoint}',
                           device=device)
    # test a single image

    imgs = glob.glob(f"{im_root}/*.png")
    imgs.sort()

    for i in range(math.ceil(len(imgs) / batch_size)):
        imgs_curr = imgs[i * batch_size:min(len(imgs), (i + 1) * batch_size)]
        result = inference_segmentor(model, imgs_curr)

        result
        import cv2
        for clz in np.unique(result[0]):
            print(clz, np.where(result[0] == clz)[0].shape)
            cv2.imwrite(f'{clz}.png', (result[0] == clz) * 255)

        from IPython import embed
        embed()


if __name__ == '__main__':
    inference_human_mask(
        '/mnt/lustre/wangwenjia/datasets/spec-mtp/images/test')
