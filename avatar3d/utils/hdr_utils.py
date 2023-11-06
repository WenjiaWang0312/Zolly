import numpy as np
import imageio
import cv2
from envmap import EnvironmentMap, rotation_matrix

hdr_path = 'abandoned_factory_canteen_02_4k.hdr'
# imageio.plugins.freeimage.download()
# img = imageio.imread(hdr_path, format='HDR-FI')
img = cv2.imread(hdr_path, flags=cv2.IMREAD_ANYDEPTH)
e = EnvironmentMap(np.array(img), 'latlong')

dcm = rotation_matrix(azimuth=np.pi / 6, elevation=np.pi / 8, roll=np.pi / 12)
crop = e.project(
    vfov=85.,  # degrees
    rotation_matrix=dcm,
    ar=4. / 3.,
    resolution=(640, 480),
    projection="perspective",
    mode="normal")

crop = np.clip(255. * crop, 0, 255).astype('uint8')
cv2.imwrite("crop.jpg", crop)
