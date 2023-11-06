import torch
import numpy as np
from pytorch3d.transforms.transform3d import Rotate, Scale, Translate


def batch_rodrigues_vectors(vec1, vec2):

    def norm_vec(vec):
        return vec / torch.norm(vec, dim=-1, keepdim=True)

    vec1 = norm_vec(vec1)
    vec2 = norm_vec(vec2)
    if torch.allclose(vec1, vec2, atol=1e-3, rtol=1e-3):
        return torch.eye(3).to(vec1.device)
    else:
        v = torch.cross(vec1, vec2, dim=-1)
        c = torch.sum(vec1 * vec2, dim=-1)
        s = torch.norm(v, dim=-1)
        skew_sym = torch.zeros((vec1.shape[0], 3, 3)).to(vec1.device)
        skew_sym[:, 0, 1] = -v[:, 2]
        skew_sym[:, 0, 2] = v[:, 1]
        skew_sym[:, 1, 2] = -v[:, 0]
        skew_sym[:, 1, 0] = v[:, 2]
        skew_sym[:, 2, 0] = -v[:, 1]
        skew_sym[:, 2, 1] = v[:, 0]
        R = torch.eye(3).to(vec1.device) + skew_sym + torch.matmul(
            skew_sym, skew_sym) * (1 - c).unsqueeze(-1) / s.unsqueeze(-1)**2
        return R

def transform_transl(transl, rotmat, scale):
    # transl: B x 3
    # rotmat: B x 3 x 3
    # scale: B x 1
    B = transl.shape[0]
    rotmat = rotmat.view(B, 3, 3)
    scale = scale.view(B, 1, 1)
    return torch.bmm(rotmat, scale * transl.unsqueeze(-1)).squeeze(-1)