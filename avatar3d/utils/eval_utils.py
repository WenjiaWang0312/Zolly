import numpy as np


def keypoint_nme(preds, targets, mask):
    N, K, _ = preds.shape
    # set mask=0 when normalize==0
    _mask = mask.copy().reshape(N, K) > 0
    distances = np.full((N, K), -1, dtype=np.float32)

    def euclid_distance(pred_kp2d, gt_kp2d):
        x = pred_kp2d[..., 0] - gt_kp2d[..., 0]
        y = pred_kp2d[..., 1] - gt_kp2d[..., 1]
        return np.sqrt(x**2 + y**2)

    # handle invalid values
    distances = euclid_distance(preds, targets).reshape(N, K) * _mask.reshape(
        N, K) + (1 - _mask) * -1
    distance_valid = distances[distances != -1]
    return distance_valid.mean() * 100

    return distances.mean()
