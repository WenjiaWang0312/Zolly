import torch

from torch import nn
from functools import partial
from einops.layers.torch import Rearrange
from zolly.models import BaseModule
from zolly.models.bert.mlp_mixer import FeedForward, PreNormResidualBN
from zolly.models.body_models.mappings import get_keypoint_num


class ZNet(nn.Module):
    '''
    Z decoder
    '''

    def __init__(
        self,
        number_of_embed=1,  # 24 + 10
        embed_dim=256,
        nhead=4,
        dim_feedforward=1024,
        numlayers=2,
    ) -> None:
        super(ZNet, self).__init__()
        self.query = nn.Embedding(number_of_embed, embed_dim)
        self.embed_dim = embed_dim
        self.conv = nn.Conv2d(1, embed_dim, kernel_size=(1, 1))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=dim_feedforward,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer,
                                             num_layers=numlayers)
        self.z_out = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, warped_d_img, n_iter=3):
        bs = warped_d_img.size(0)  # B, 1, 56, 56
        memory = self.conv(warped_d_img)  # B, 256, 56, 56
        decoder_out = self.decoder(
            self.query.weight.unsqueeze(0).repeat(bs, 1, 1),  # B, 1, 256
            memory.reshape(bs, self.embed_dim, -1).permute(0, 2,
                                                           1))  # B, 12544, 256
        # B, 1, 256
        z = self.z_out(decoder_out.reshape(bs, -1))  # B, 1
        z = self.sigmoid(z) * 10
        return {'pred_z': z, 'warped_d_feat': memory}


class DistortionTransformer(nn.Module):
    '''
    Z decoder
    '''

    def __init__(
        self,
        in_channels=1,
        number_of_embed=1,
        embed_dim=256,
        nhead=4,
        dim_feedforward=1024,
        numlayers=2,
    ) -> None:
        super(DistortionTransformer, self).__init__()
        self.query = nn.Embedding(number_of_embed, embed_dim)
        self.embed_dim = embed_dim
        self.fc = nn.Linear(in_channels, embed_dim)
        self.fc2 = nn.Linear(embed_dim, 1)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=dim_feedforward,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer,
                                             num_layers=numlayers)

        self.sigmoid = nn.Sigmoid()

    def forward(self, distortion_feature):
        bs = distortion_feature.size(0)  # B, 431, C
        memory = self.fc(distortion_feature)  # B, 431, embed_dim
        decoder_out = self.decoder(
            self.query.weight.unsqueeze(0).repeat(bs, 1, 1),  # B, 1, embed_dim
            memory)  # B, 1, embed_dim
        # B, 431, embed_dim
        decoder_out = self.fc2(decoder_out)
        pred_z = self.sigmoid(decoder_out.squeeze(-1)) * 10
        return pred_z


class ZollyHead(BaseModule):
    '''
    End-to-end Graphormer network for human pose and mesh reconstruction from a single image.
    '''

    def __init__(
        self,
        feature_size=56,
        convention='h36m',
        in_channels=256,
        cam_feat_dim=2048,
        pred_kp3d=False,
        ncam=3,
        nhdim=512,
        nhdim_cam=250,
        nverts_sub2=431,
        nverts_sub1=1723,
        nverts=6890,
        znet_config=dict(
            number_of_embed=1,  # 24 + 10
            embed_dim=256,
            nhead=4,
            dim_feedforward=1024,
            numlayers=2,
        ),
        init_cfg=None,
    ):
        super(ZollyHead, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.feature_size = feature_size

        self.upsampling = torch.nn.Linear(nverts_sub2, nverts_sub1)
        self.upsampling2 = torch.nn.Linear(nverts_sub1, nverts)
        self.cam_enc = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=in_channels * 2,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels * 2,
                      out_channels=in_channels * 4,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(in_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels * 4,
                      out_channels=in_channels * 8,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(in_channels * 8),
            nn.ReLU(inplace=True),
        )
        self.cam_dec = nn.Sequential(torch.nn.Linear(cam_feat_dim, nhdim_cam),
                                     torch.nn.Linear(nhdim_cam, nhdim_cam),
                                     torch.nn.Linear(nhdim_cam, ncam))
        self.decoder = self.MLPMixer(in_channels=self.in_channels, dim=nhdim)
        num_joints = get_keypoint_num(convention)
        self.convention = convention

        self.dec_z = ZNet(**znet_config)
        # self.conv_d_feat = Conv1x1(znet_config['embed_dim'], 128)
        if pred_kp3d:
            self.vertices2joints = torch.nn.Conv1d(nverts_sub2,
                                                   num_joints,
                                                   kernel_size=1)
        self.pred_kp3d = pred_kp3d
        # self.merger_pred_2 = torch.nn.Linear(nhdim + self.dec_z.embed_dim,
        #                                      nverts_sub2)
        self.merger_pred_2 = torch.nn.Linear(nhdim, nverts_sub2)
        self.after_merger = torch.nn.Linear(nhdim + self.dec_z.embed_dim, 3)
        # self.after_merger = torch.nn.Linear(nhdim, 3)

    def MLPMixer(self,
                 dim=512,
                 in_channels=48,
                 expansion_factor=2,
                 expansion_factor_token=1,
                 dropout=0.):
        channels = in_channels

        patch_size = 2
        image_h, image_w = self.feature_size, self.feature_size
        num_vertices = 431
        depth = 3
        num_patches = (image_h // patch_size) * (image_w // patch_size)
        chan_first, chan_last = partial(nn.Conv1d,
                                        kernel_size=1), partial(nn.Conv1d,
                                                                kernel_size=7,
                                                                padding=3,
                                                                groups=dim)
        return nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_size,
                      p2=patch_size),
            nn.Linear((patch_size**2) * channels, dim),
            #PreNormResidual(dim, FeedForward(num_patches, num_vertices, expansion_factor, dropout, chan_first)),
            FeedForward(num_patches, num_vertices, expansion_factor, dropout,
                        chan_first),
            Rearrange('b n c -> b c n'),
            *[
                nn.Sequential(
                    PreNormResidualBN(
                        dim,
                        FeedForward(dim, dim, expansion_factor, dropout,
                                    chan_first)),
                    PreNormResidualBN(
                        dim,
                        FeedForward(dim, dim, expansion_factor_token, dropout,
                                    chan_last)),
                ) for _ in range(depth)
            ],
            # nn.LayerNorm(dim),
            # nn.Linear(dim, 3)
        )

    def vertex_sample(self, feature, vertex2d):  #B, C, H, W
        res = feature.shape[-1]
        vertex2d = (vertex2d[..., :2] * res).long()
        sampled_feature = feature[:, :, vertex2d[:, 1], vertex2d[:, 0]]
        return sampled_feature.permute(0, 2, 1)  #B, 431, C

    def forward(
            self,
            x,  # B, 256, 56, 56
            warped_d_img,  # B, 1, 56, 56
            vertex_uv,  # 1, 431, 2
    ):

        v_feat = x
        vertex_uv = vertex_uv.to(v_feat.device)
        dec_z_out = self.dec_z(warped_d_img)
        warped_d_feat = dec_z_out['warped_d_feat']  # B, C2, 56, 56
        pred_z = dec_z_out['pred_z']

        vertex_distortion_feature = self.vertex_sample(warped_d_feat,
                                                       vertex_uv)  # B,  431, C

        features = self.decoder(v_feat)  # B, 512, 431
        features = features.transpose(1, 2)  # B, 431, 512

        merger_2 = self.merger_pred_2(features).transpose(1, 2).softmax(
            2)  # # B, 431, 431, -> B, 431, 431
        features = merger_2 @ features  # B, 431, 431 @ B, 431, nhdim -> B, 431, ndim=512

        features = torch.cat([features, vertex_distortion_feature],
                             -1)  # B, 431, 512+256

        pred_vertices_sub2 = self.after_merger(features)  # B, 431, 3
        if self.pred_kp3d:
            pred_keypoints3d = self.vertices2joints(
                pred_vertices_sub2)  # B, 14, 3
        cam_feat = self.cam_enc(v_feat)
        cam_feat = cam_feat.mean(-1).mean(-1)
        pred_cam = self.cam_dec(cam_feat)

        temp_transpose = pred_vertices_sub2.transpose(1, 2)  # B, 3, 431
        pred_vertices_sub = self.upsampling(temp_transpose)  # B, 3, 1723
        pred_vertices_full = self.upsampling2(pred_vertices_sub)  # B, 3, 6890
        pred_vertices_sub = pred_vertices_sub.transpose(1, 2)  # B, 1723, 3
        pred_vertices_full = pred_vertices_full.transpose(1, 2)  # B, 6890, 3

        output = dict(pred_cam=pred_cam,
                      pred_vertices=pred_vertices_full,
                      pred_vertices_sub1=pred_vertices_sub,
                      pred_vertices_sub2=pred_vertices_sub2,
                      pred_z=pred_z)

        if self.pred_kp3d:
            output.update(dict(pred_keypoints3d=pred_keypoints3d))

        return output
