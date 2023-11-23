"""
This file provides a wrapper around GraphCNN and SMPLParamRegressor and is useful for inference since it fuses both forward passes in one.
It returns both the non-parametric and parametric shapes, as well as the camera and the regressed SMPL parameters.
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

from zolly.models import BaseModule
from zolly.models.layers.graph_cnn import GraphCNN
from zolly.structures.meshes.utils import MeshSampler


class FCBlock(nn.Module):
    """Wrapper around nn.Linear that includes batch normalization and activation functions."""

    def __init__(self,
                 in_size,
                 out_size,
                 batchnorm=True,
                 activation=nn.ReLU(inplace=True),
                 dropout=False):
        super(FCBlock, self).__init__()
        module_list = [nn.Linear(in_size, out_size)]
        if batchnorm:
            module_list.append(nn.BatchNorm1d(out_size))
        if activation is not None:
            module_list.append(activation)
        if dropout:
            module_list.append(dropout)
        self.fc_block = nn.Sequential(*module_list)

    def forward(self, x):
        return self.fc_block(x)


class FCResBlock(nn.Module):
    """Residual block using fully-connected layers."""

    def __init__(self,
                 in_size,
                 out_size,
                 batchnorm=True,
                 activation=nn.ReLU(inplace=True),
                 dropout=False):
        super(FCResBlock, self).__init__()
        self.fc_block = nn.Sequential(nn.Linear(in_size, out_size),
                                      nn.BatchNorm1d(out_size),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(out_size, out_size),
                                      nn.BatchNorm1d(out_size))

    def forward(self, x):
        return F.relu(x + self.fc_block(x))


def batch_svd(A):
    """Wrapper around torch.svd that works when the input is a batch of matrices."""
    U_list = []
    S_list = []
    V_list = []
    for i in range(A.shape[0]):
        U, S, V = torch.svd(A[i])
        U_list.append(U)
        S_list.append(S)
        V_list.append(V)
    U = torch.stack(U_list, dim=0)
    S = torch.stack(S_list, dim=0)
    V = torch.stack(V_list, dim=0)
    return U, S, V


class SMPLParamRegressor(nn.Module):

    def __init__(self, use_cpu_svd=True):
        super(SMPLParamRegressor, self).__init__()
        # 1723 is the number of vertices in the subsampled SMPL mesh
        self.layers = nn.Sequential(FCBlock(1723 * 6, 1024),
                                    FCResBlock(1024, 1024),
                                    FCResBlock(1024, 1024),
                                    nn.Linear(1024, 24 * 3 * 3 + 10))
        self.use_cpu_svd = use_cpu_svd

    def forward(self, x):
        """Forward pass.
        Input:
            x: size = (B, 1723*6)
        Returns:
            SMPL pose parameters as rotation matrices: size = (B,24,3,3)
            SMPL shape parameters: size = (B,10)
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.layers(x)
        rotmat = x[:, :24 * 3 * 3].view(-1, 24, 3, 3).contiguous()
        betas = x[:, 24 * 3 * 3:].contiguous()
        rotmat = rotmat.view(-1, 3, 3).contiguous()
        orig_device = rotmat.device
        if self.use_cpu_svd:
            rotmat = rotmat.cpu()
        U, S, V = batch_svd(rotmat)

        rotmat = torch.matmul(U, V.transpose(1, 2))
        det = torch.zeros(rotmat.shape[0], 1, 1).to(rotmat.device)
        with torch.no_grad():
            for i in range(rotmat.shape[0]):
                det[i] = torch.det(rotmat[i])
        rotmat = rotmat * det
        rotmat = rotmat.view(batch_size, 24, 3, 3)
        rotmat = rotmat.to(orig_device)
        return rotmat, betas


class CMRHead(BaseModule):

    def __init__(self, mesh_sampler, num_layers, num_channels, init_cfg=None):
        super(CMRHead, self).__init__(init_cfg=init_cfg)
        mesh_sampler = MeshSampler(**mesh_sampler)
        self.graph_cnn = GraphCNN(mesh_sampler.adjmat,
                                  mesh_sampler.ref_vertices.t(), num_layers,
                                  num_channels)
        self.mesh_sampler = mesh_sampler
        self.smpl_param_regressor = SMPLParamRegressor()

    def forward(self,
                x,
                train_graph_cnn=True,
                train_smpl_param_regressor=True,
                detach=False):
        """Fused forward pass for the 2 networks
        Inputs:
            x: size = (B, 2048, 7, 7)
        Returns:
            Regressed non-parametric shape: size = (B, 6890, 3)
            Regressed SMPL shape: size = (B, 6890, 3)
            Weak-perspective camera: size = (B, 3)
            SMPL pose parameters (as rotation matrices): size = (B, 24, 3, 3)
            SMPL shape parameters: size = (B, 10)
        """
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[-1]
        batch_size = x.shape[0]
        if not train_graph_cnn:
            with torch.no_grad():
                pred_vertices_sub1, pred_cam = self.graph_cnn(x)
        else:
            pred_vertices_sub1, pred_cam = self.graph_cnn(x)

        if detach:
            x = pred_vertices_sub1.transpose(1, 2).detach()
        else:
            x = pred_vertices_sub1.transpose(1, 2)

        ref_vertices = self.mesh_sampler.ref_vertices[None, :, :].expand(
            batch_size, -1, -1).to(x.device)
        x = torch.cat([x, ref_vertices], dim=-1)
        if not train_smpl_param_regressor:
            with torch.no_grad():
                pred_rotmat, pred_betas = self.smpl_param_regressor(x)
        else:
            pred_rotmat, pred_betas = self.smpl_param_regressor(x)
        output = dict(pred_cam=pred_cam,
                      pred_pose=pred_rotmat,
                      pred_shape=pred_betas,
                      pred_vertices_sub1=pred_vertices_sub1.permute(0, 2, 1))
        return output
