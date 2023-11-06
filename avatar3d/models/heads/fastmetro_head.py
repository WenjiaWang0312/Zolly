import torch

from torch import nn
from avatar3d.models import BaseModule
from avatar3d.models.body_models.mappings import get_keypoint_num
import numpy as np
from avatar3d.models.transformer.transformer import build_transformer
from avatar3d.models.transformer.position_encoding import build_position_encoding


class FastMetroHead(BaseModule):
    '''
    End-to-end Graphormer network for human pose and mesh reconstruction from a single image.
    '''

    def __init__(self,
                 convention='h36m',
                 num_enc_layers=1,
                 num_dec_layers=1,
                 model_dim_1=512,
                 model_dim_2=128,
                 feedforward_dim_1=2048,
                 feedforward_dim_2=512,
                 conv_1x1_dim=2048,
                 transformer_dropout=0.1,
                 transformer_nhead=8,
                 pos_type='sine',
                 num_vertices=431,
                 adjmat_indices=None,
                 adjmat_values=None,
                 adjmat_size=None,
                 init_cfg=None):
        super(FastMetroHead, self).__init__(init_cfg=init_cfg)

        num_joints = get_keypoint_num(convention)
        self.num_joints = num_joints
        self.num_vertices = num_vertices

        # configurations for the first transformer
        self.transformer_config_1 = {
            "model_dim": model_dim_1,
            "dropout": transformer_dropout,
            "nhead": transformer_nhead,
            "feedforward_dim": feedforward_dim_1,
            "num_enc_layers": num_enc_layers,
            "num_dec_layers": num_dec_layers,
            "pos_type": pos_type
        }
        # configurations for the second transformer
        self.transformer_config_2 = {
            "model_dim": model_dim_2,
            "dropout": transformer_dropout,
            "nhead": transformer_nhead,
            "feedforward_dim": feedforward_dim_2,
            "num_enc_layers": num_enc_layers,
            "num_dec_layers": num_dec_layers,
            "pos_type": pos_type
        }

        # build transformers
        self.transformer_1 = build_transformer(self.transformer_config_1)
        self.transformer_2 = build_transformer(self.transformer_config_2)
        # dimensionality reduction
        self.dim_reduce_enc_cam = nn.Linear(
            self.transformer_config_1["model_dim"],
            self.transformer_config_2["model_dim"])
        self.dim_reduce_enc_img = nn.Linear(
            self.transformer_config_1["model_dim"],
            self.transformer_config_2["model_dim"])
        self.dim_reduce_dec = nn.Linear(self.transformer_config_1["model_dim"],
                                        self.transformer_config_2["model_dim"])

        # token embeddings
        self.cam_token_embed = nn.Embedding(
            1, self.transformer_config_1["model_dim"])
        self.joint_token_embed = nn.Embedding(
            self.num_joints, self.transformer_config_1["model_dim"])
        self.vertex_token_embed = nn.Embedding(
            self.num_vertices, self.transformer_config_1["model_dim"])
        # positional encodings
        self.position_encoding_1 = build_position_encoding(
            pos_type=self.transformer_config_1['pos_type'],
            hidden_dim=self.transformer_config_1['model_dim'])
        self.position_encoding_2 = build_position_encoding(
            pos_type=self.transformer_config_2['pos_type'],
            hidden_dim=self.transformer_config_2['model_dim'])
        # estimators
        self.xyz_regressor = nn.Linear(self.transformer_config_2["model_dim"],
                                       3)
        self.cam_predictor = nn.Linear(self.transformer_config_2["model_dim"],
                                       3)

        # 1x1 Convolution
        self.conv_1x1 = nn.Conv2d(conv_1x1_dim,
                                  self.transformer_config_1["model_dim"],
                                  kernel_size=1)

        # attention mask
        zeros_1 = torch.tensor(
            np.zeros((num_vertices, num_joints)).astype(bool))
        zeros_2 = torch.tensor(
            np.zeros((num_joints, (num_joints + num_vertices))).astype(bool))

        adjacency_indices = torch.load(adjmat_indices)
        adjacency_matrix_value = torch.load(adjmat_values)
        adjacency_matrix_size = torch.load(adjmat_size)

        adjacency_matrix = torch.sparse_coo_tensor(
            adjacency_indices,
            adjacency_matrix_value,
            size=adjacency_matrix_size).to_dense()

        temp_mask_1 = (adjacency_matrix == 0)
        temp_mask_2 = torch.cat([zeros_1, temp_mask_1], dim=1)
        self.attention_mask = torch.cat([zeros_2, temp_mask_2], dim=0)

        # learnable upsampling layer is used (from coarse mesh to intermediate mesh); for visually pleasing mesh result
        ### pre-computed upsampling matrix is used (from intermediate mesh to fine mesh); to reduce optimization difficulty
        self.coarse2intermediate_upsample = nn.Linear(431, 1723)

    def forward(self, grid_feat):
        # grid_feat: B, 48, 56, 56 for hrnet, B, 2048, 7, 7 for resnet
        if isinstance(grid_feat, (tuple, list)):
            grid_feat = grid_feat[0]

        device = grid_feat.device
        batch_size = grid_feat.size(0)

        # preparation
        cam_token = self.cam_token_embed.weight.unsqueeze(1).repeat(
            1, batch_size, 1)  # 1 X batch_size X 512
        jv_tokens = torch.cat(
            [self.joint_token_embed.weight, self.vertex_token_embed.weight],
            dim=0).unsqueeze(1).repeat(
                1, batch_size,
                1)  # (num_joints + num_vertices) X batch_size X 512
        attention_mask = self.attention_mask.to(
            device
        )  # (num_joints + num_vertices) X (num_joints + num_vertices)

        _, _, h, w = grid_feat.shape
        grid_feat = self.conv_1x1(grid_feat).flatten(2).permute(
            2, 0, 1)  # 49 X batch_size X 512

        # positional encodings
        pos_enc_1 = self.position_encoding_1(batch_size, h, w,
                                             device).flatten(2).permute(
                                                 2, 0,
                                                 1)  # 49 X batch_size X 512
        pos_enc_2 = self.position_encoding_2(batch_size, h, w,
                                             device).flatten(2).permute(
                                                 2, 0,
                                                 1)  # 49 X batch_size X 128

        # first transformer encoder-decoder
        cam_features_1, enc_img_features_1, jv_features_1 = self.transformer_1(
            grid_feat,
            cam_token,
            jv_tokens,
            pos_enc_1,
            attention_mask=attention_mask)

        # progressive dimensionality reduction
        reduced_cam_features_1 = self.dim_reduce_enc_cam(
            cam_features_1)  # 1 X batch_size X 128
        reduced_enc_img_features_1 = self.dim_reduce_enc_img(
            enc_img_features_1)  # 49 X batch_size X 128
        reduced_jv_features_1 = self.dim_reduce_dec(
            jv_features_1)  # (num_joints + num_vertices) X batch_size X 128

        # second transformer encoder-decoder
        cam_features_2, _, jv_features_2 = self.transformer_2(
            reduced_enc_img_features_1,
            reduced_cam_features_1,
            reduced_jv_features_1,
            pos_enc_2,
            attention_mask=attention_mask)

        # estimators
        pred_cam = self.cam_predictor(cam_features_2).view(batch_size,
                                                           3)  # batch_size X 3
        pred_3d_coordinates = self.xyz_regressor(jv_features_2.transpose(
            0, 1))  # batch_size X (num_joints + num_vertices) X 3
        pred_3d_joints = pred_3d_coordinates[:, :self.
                                             num_joints, :]  # batch_size X num_joints X 3
        pred_3d_vertices_coarse = pred_3d_coordinates[:, self.
                                                      num_joints:, :]  # batch_size X num_vertices(coarse) X 3

        # coarse-to-intermediate mesh upsampling
        pred_3d_vertices_intermediate = self.coarse2intermediate_upsample(
            pred_3d_vertices_coarse.transpose(1, 2)).transpose(
                1, 2)  # batch_size X num_vertices(intermediate) X 3
        # intermediate-to-fine mesh upsampling

        output = dict(pred_cam=pred_cam,
                      pred_keypoints3d=pred_3d_joints,
                      pred_vertices_sub1=pred_3d_vertices_intermediate,
                      pred_vertices_sub2=pred_3d_vertices_coarse)
        return output
