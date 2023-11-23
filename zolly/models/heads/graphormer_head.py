import torch

from torch import nn
from functools import partial
from einops.layers.torch import Rearrange
from pytorch_transformers.modeling_bert import BertConfig
from zolly.models import BaseModule
from zolly.models.bert.mlp_mixer import FeedForward, PreNormResidualBN
from zolly.models.body_models.mappings import get_keypoint_num

from zolly.models.bert.graphormer import Graphormer
from zolly.structures.meshes.utils import MeshSampler


class GraphormerHead(BaseModule):
    '''
    End-to-end Graphormer network for human pose and mesh reconstruction from a single image.
    '''

    def __init__(self,
                 convention='h36m',
                 in_channel=48,
                 ncam=3,
                 nhdim=512,
                 nhdim_cam=250,
                 nverts_sub2=431,
                 nverts_sub1=1723,
                 nverts=6890,
                 init_cfg=None):
        super(GraphormerHead, self).__init__(init_cfg=init_cfg)

        self.upsampling = torch.nn.Linear(nverts_sub2, nverts_sub1)
        self.upsampling2 = torch.nn.Linear(nverts_sub1, nverts)
        self.cam_dec = nn.Sequential(torch.nn.Linear(3, 1),
                                     Rearrange('b nsub2 c -> b c nsub2', c=1),
                                     torch.nn.Linear(nverts_sub2, nhdim_cam),
                                     torch.nn.Linear(nhdim_cam, ncam),
                                     Rearrange('b c ncam -> b ncam c', c=1),
                                     Rearrange('b ncam c -> b (ncam c)', c=1))
        # self.cam_param_fc = torch.nn.Linear(3, 1)
        # self.cam_param_fc2 = torch.nn.Linear(nverts_sub2, nhdim_cam)
        # self.cam_param_fc3 = torch.nn.Linear(nhdim_cam, ncam)
        self.decoder = self.MLPMixer(in_channel=in_channel, dim=nhdim)
        num_joints = get_keypoint_num(convention)
        self.convention = convention

        self.vertices2joints = torch.nn.Conv1d(nverts_sub2,
                                               num_joints,
                                               kernel_size=1)
        self.merger_pred_2 = torch.nn.Linear(nhdim, nverts_sub2)
        self.after_merger = torch.nn.Linear(nhdim, 3)

    def MLPMixer(self,
                 dim=512,
                 in_channel=48,
                 expansion_factor=2,
                 expansion_factor_token=1,
                 dropout=0.):
        channels = in_channel

        patch_size = 2
        image_h, image_w = 56, 56
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

    def forward(self, grid_feat):
        # grid_feat: B, 48, 56, 56
        if isinstance(grid_feat, (tuple, list)):
            grid_feat = grid_feat[0]
        features = self.decoder(grid_feat)  # B, 512, N=431

        features = features.transpose(1, 2)  # B, N, 512
        merger_2 = self.merger_pred_2(features).transpose(1, 2).softmax(
            2)  # # B, 431, N, -> B, 431, N
        features = merger_2 @ features  # B, 431, N @ B, N, nhdim -> B, 431, ndim
        features = self.after_merger(features)  # B, 431, 3

        pred_vertices_sub2 = features  # B, 431, 3
        pred_keypoints3d = self.vertices2joints(pred_vertices_sub2)  # B, 14, 3

        # learn camera parameters
        pred_cam = self.cam_dec(pred_vertices_sub2)
        # x = self.cam_param_fc(pred_vertices_sub2)  # B, 431, 1
        # x = x.transpose(1, 2)  # B, 1, 431
        # x = self.cam_param_fc2(x)  # B, 1, 250
        # x = self.cam_param_fc3(x)  # B, 1, 3
        # cam_param = x.transpose(1, 2)  # B, 3, 1
        # pred_cam = cam_param.squeeze()  # B, 3

        temp_transpose = pred_vertices_sub2.transpose(1, 2)  # B, 3, 431
        pred_vertices_sub = self.upsampling(temp_transpose)  # B, 3, 1723
        pred_vertices_full = self.upsampling2(pred_vertices_sub)  # B, 3, 6890
        pred_vertices_sub = pred_vertices_sub.transpose(1, 2)  # B, 1723, 3
        pred_vertices_full = pred_vertices_full.transpose(1, 2)  # B, 6890, 3
        output = dict(pred_cam=pred_cam,
                      pred_keypoints3d=pred_keypoints3d,
                      pred_vertices=pred_vertices_full,
                      pred_vertices_sub1=pred_vertices_sub,
                      pred_vertices_sub2=pred_vertices_sub2)

        return output


class GraphormerHead_orig(BaseModule):
    '''
    End-to-end Graphormer network for human pose and mesh reconstruction from a single image.
    '''

    def __init__(self, config_path, mesh_sampler, init_cfg=None):
        super(GraphormerHead_orig, self).__init__(init_cfg=init_cfg)

        config = BertConfig.from_pretrained(config_path)

        input_feat_dim = [2051, 512, 128]
        output_feat_dim = input_feat_dim[1:] + [3]
        config.output_attentions = False
        config.hidden_dropout_prob = 0.1
        which_blk_graph = [0, 0, 1]
        for i in range(len(output_feat_dim)):
            config.img_feature_dim = input_feat_dim[i]
            config.output_feature_dim = output_feat_dim[i]

            if which_blk_graph[i] == 1:
                config.graph_conv = True
            else:
                config.graph_conv = False
        config.mesh_type = 'body'

        self.mesh_sampler = MeshSampler(**mesh_sampler, convention='lsp')
        config.mesh_sampler = self.mesh_sampler
        self.trans_encoder = Graphormer(config)
        self.config = config
        self.upsampling = torch.nn.Linear(431, 1723)
        self.upsampling2 = torch.nn.Linear(1723, 6890)
        self.cam_param_fc = torch.nn.Linear(3, 1)
        self.cam_param_fc2 = torch.nn.Linear(431, 250)
        self.cam_param_fc3 = torch.nn.Linear(250, 3)
        self.grid_feat_dim = torch.nn.Linear(1024, 2051)

    def forward(self, feats, meta_masks=None, is_train=False):
        image_feat, grid_feat = feats
        num_joints = 14
        batch_size = image_feat.size(0)
        # Generate T-pose template mesh
        ref_vertices = self.mesh_sampler.ref_vertices[None, :, :].expand(
            batch_size, -1, -1).to(image_feat.device)
        ref_joints = self.mesh_sampler.ref_joints.expand(
            batch_size, -1, -1).to(image_feat.device)
        ref_vertices = torch.cat([ref_joints, ref_vertices], 1)
        # concatinate image feat and 3d mesh template
        image_feat = image_feat.view(batch_size, 1,
                                     2048).expand(-1, ref_vertices.shape[-2],
                                                  -1)
        # process grid features
        grid_feat = torch.flatten(grid_feat, start_dim=2)
        grid_feat = grid_feat.transpose(1, 2)

        grid_feat = self.grid_feat_dim(grid_feat)
        # concatinate image feat and template mesh to form the joint/vertex queries
        features = torch.cat([ref_vertices, image_feat], dim=2)
        # prepare input tokens including joint/vertex queries and grid features
        features = torch.cat([features, grid_feat], dim=1)

        from IPython import embed
        embed()

        # if is_train == True:
        #     # apply mask vertex/joint modeling
        #     # meta_masks is a tensor of all the masks, randomly generated in dataloader
        #     # we pre-define a [MASK] token, which is a floating-value vector with 0.01s
        #     special_token = torch.ones_like(features[:, :-49, :]).cuda() * 0.01
        #     features[:, :
        #              -49, :] = features[:, :
        #                                 -49, :] * meta_masks + special_token * (
        #                                     1 - meta_masks)

        # forward pass
        if self.config.output_attentions == True:
            features, hidden_states, att = self.trans_encoder(features)
        else:
            features = self.trans_encoder(features)

        pred_keypoints3d = features[:, :num_joints, :]
        pred_vertices_sub2 = features[:, num_joints:-49, :]

        # learn camera parameters
        x = self.cam_param_fc(pred_vertices_sub2)
        x = x.transpose(1, 2)
        x = self.cam_param_fc2(x)
        x = self.cam_param_fc3(x)
        pred_cam = x.transpose(1, 2)
        pred_cam = pred_cam.squeeze()

        temp_transpose = pred_vertices_sub2.transpose(1, 2)
        pred_vertices_sub = self.upsampling(temp_transpose)
        pred_vertices_full = self.upsampling2(pred_vertices_sub)
        pred_vertices_sub = pred_vertices_sub.transpose(1, 2)
        pred_vertices_full = pred_vertices_full.transpose(1, 2)

        output = dict(pred_cam=pred_cam,
                      pred_keypoints3d=pred_keypoints3d,
                      pred_vertices=pred_vertices_full,
                      pred_vertices_sub1=pred_vertices_sub,
                      pred_vertices_sub2=pred_vertices_sub2)
        return output
