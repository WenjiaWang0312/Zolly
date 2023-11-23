import itertools
import numpy as np
import torch
import torch.nn as nn

from zolly.utils.keypoint_utils import search_limbs


class HingeLoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 max_bound=None,
                 min_bound=None):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.max_bound = max_bound
        self.min_bound = min_bound

    def forward(self,
                pred,
                loss_weight_override=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)
        loss_weight = (loss_weight_override if loss_weight_override is not None
                       else self.loss_weight)

        loss = 0.
        if self.min_bound is not None:
            mask = (pred < self.min_bound)
            loss += (self.min_bound - pred)**2 * mask
        if self.max_bound is not None:
            mask = (pred > self.max_bound)
            loss += (self.max_bound - pred)**2 * mask

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        loss *= loss_weight

        return loss


class LimbLengthLoss(nn.Module):
    """Limb length loss for body shape parameters. As betas are associated with
    the height of a person, fitting on limb length help determine body shape
    parameters. It penalizes the L2 distance between target limb length and
    pred limb length. Note that it should take keypoints3d as input, as limb
    length computed from keypoints2d varies with camera.

    Args:
        convention (str): Limb convention to search for keypoint connections.
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
        eps (float, optional): epsilon for computing normalized limb vector.
            Defaults to 1e-4.
    """

    def __init__(self,
                 convention,
                 dim=3,
                 reduction='mean',
                 loss_weight=1.0,
                 eps=1e-4,
                 conf_threshold=1.0):
        super().__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps
        self.dim = dim
        self.convention = convention
        self.conf_threshold = conf_threshold
        self.get_limb_idxs(convention)

    def get_limb_idxs(self, convention):
        self.convention = convention
        limb_idxs, _ = search_limbs(data_source=convention)
        if convention == 'mano_right':
            limb_idxs = sorted(limb_idxs['right_hand'])
        elif convention == 'mano_left':
            limb_idxs = sorted(limb_idxs['left_hand'])
        elif convention == 'mano_full':
            limb_idxs = sorted(limb_idxs['left_hand'] +
                               limb_idxs['right_hand'])
        else:
            limb_idxs = sorted(limb_idxs['body'])
        self.limb_idxs = np.array(
            list(x for x, _ in itertools.groupby(limb_idxs)))

    def _compute_limb_length(self, keypoints):
        kp_src = keypoints[:, self.limb_idxs[:, 0], :self.dim]
        kp_dst = keypoints[:, self.limb_idxs[:, 1], :self.dim]
        limb_vec = kp_dst - kp_src
        limb_length = torch.norm(limb_vec, dim=2)
        return limb_length

    def _keypoint_conf_to_limb_conf(self, keypoint_conf):
        keypoint_conf = (keypoint_conf > self.conf_threshold) * keypoint_conf
        limb_conf = torch.min(keypoint_conf[:, self.limb_idxs[:, 1]],
                              keypoint_conf[:, self.limb_idxs[:, 0]])
        return limb_conf

    def forward(self,
                pred,
                target,
                convention=None,
                pred_conf=None,
                target_conf=None,
                loss_weight_override=None,
                reduction_override=None):
        """Forward function of LimbLengthLoss.

        Args:
            pred (torch.Tensor): The predicted smpl keypoints3d.
                Shape should be (N, K, 3).
                B: batch size. K: number of keypoints.
            target (torch.Tensor): The ground-truth keypoints3d.
                Shape should be (N, K, 3).
            pred_conf (torch.Tensor, optional): Confidence of
                predicted keypoints. Shape should be (N, K).
            target_conf (torch.Tensor, optional): Confidence of
                target keypoints. Shape should be (N, K).
            loss_weight_override (float, optional): The weight of loss used to
                override the original weight of loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None
        Returns:
            torch.Tensor: The calculated loss
        """
        if isinstance(convention, str):
            if convention != self.convention:
                self.get_limb_idxs(convention)
        assert pred.dim() == 3 and pred.shape[-1] == self.dim
        assert pred.shape == target.shape
        if pred_conf is not None:
            assert pred_conf.dim() == 3
            assert pred_conf.shape[:2] == pred.shape[:2]
        if target_conf is not None:
            assert target_conf.dim() == 3
            assert target_conf.shape[:2] == target.shape[:2]
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)
        loss_weight = (loss_weight_override if loss_weight_override is not None
                       else self.loss_weight)

        limb_len_target = self._compute_limb_length(target)
        limb_len_pred = self._compute_limb_length(pred)

        if target_conf is None:
            target_conf = torch.ones_like(target[..., 0])
        if pred_conf is None:
            pred_conf = torch.ones_like(pred[..., 0])
        limb_conf_target = self._keypoint_conf_to_limb_conf(target_conf)
        limb_conf_pred = self._keypoint_conf_to_limb_conf(pred_conf)

        limb_conf = limb_conf_target.squeeze(2) * limb_conf_pred
        diff_len = limb_len_target - limb_len_pred
        loss = diff_len**2 * limb_conf

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        loss *= loss_weight

        return loss


class LimbDirectionLoss(LimbLengthLoss):

    def __init__(self,
                 convention='',
                 dim=3,
                 reduction='mean',
                 loss_weight=1,
                 eps=1e-6,
                 conf_threshold=1.0):
        super().__init__(convention, dim, reduction, loss_weight, eps,
                         conf_threshold)

    def _compute_limb_vector(self, keypoints):
        kp_src = keypoints[:, self.limb_idxs[:, 0], :self.dim]
        kp_dst = keypoints[:, self.limb_idxs[:, 1], :self.dim]
        limb_vec = kp_dst - kp_src
        limb_length = torch.norm(limb_vec, dim=2)
        return limb_vec / (limb_length.unsqueeze(-1) + self.eps)

    def forward(self,
                pred,
                target,
                convention=None,
                pred_conf=None,
                target_conf=None,
                loss_weight_override=None,
                reduction_override=None):
        """Forward function of LimbLengthLoss.

        Args:
            pred (torch.Tensor): The predicted smpl keypoints3d.
                Shape should be (N, K, 3).
                B: batch size. K: number of keypoints.
            target (torch.Tensor): The ground-truth keypoints3d.
                Shape should be (N, K, 3).
            pred_conf (torch.Tensor, optional): Confidence of
                predicted keypoints. Shape should be (N, K).
            target_conf (torch.Tensor, optional): Confidence of
                target keypoints. Shape should be (N, K).
            loss_weight_override (float, optional): The weight of loss used to
                override the original weight of loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None
        Returns:
            torch.Tensor: The calculated loss
        """
        if isinstance(convention, str):
            if convention != self.convention:
                self.get_limb_idxs(convention)
        assert pred.dim() == 3 and pred.shape[-1] == self.dim
        assert pred.shape == target.shape
        if pred_conf is not None:
            assert pred_conf.dim() == 3
            assert pred_conf.shape[:2] == pred.shape[:2]
        if target_conf is not None:
            assert target_conf.dim() == 3
            assert target_conf.shape[:2] == target.shape[:2]
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)
        loss_weight = (loss_weight_override if loss_weight_override is not None
                       else self.loss_weight)

        limb_vec_target = self._compute_limb_vector(target)
        limb_vec_pred = self._compute_limb_vector(pred)

        if target_conf is None:
            target_conf = torch.ones_like(target[..., 0])
        if pred_conf is None:
            pred_conf = torch.ones_like(pred[..., 0])
        limb_conf_target = self._keypoint_conf_to_limb_conf(target_conf)
        limb_conf_pred = self._keypoint_conf_to_limb_conf(pred_conf)
        limb_conf = limb_conf_target * limb_conf_pred

        limb_cosine = torch.sum(limb_vec_target * limb_vec_pred, -1)
        loss = (limb_cosine - 1)**2 * limb_conf

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        loss *= loss_weight

        return loss
