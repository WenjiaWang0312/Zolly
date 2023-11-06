from configs.prompt.base import (root, convention, body_model_train,
                                 body_model_test, pw3d_train_transl, test_dict)

_base_ = ['../base.py']
use_adversarial_train = True

checkpoint_config = dict(interval=2, )
# evaluate
evaluation = dict(metric=['pa-mpjpe', 'mpjpe'])

optimizer = dict(
    backbone=dict(type='Adam', lr=2e-4, betas=(0.9, 0.999), weight_decay=0),
    neck=dict(type='Adam', lr=2e-4, betas=(0.9, 0.999), weight_decay=0),
    verts_head=dict(type='Adam', lr=2e-4, betas=(0.9, 0.999), weight_decay=0),
    iuvd_head=dict(type='Adam', lr=2e-4, betas=(0.9, 0.999), weight_decay=0),
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='Fixed', by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=190)

log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

work_dirs = '/mnt/lustre/work_dirs/'

checkpoint = None
uv_res = 56
uv_renderer = dict(
    type='UVRenderer',
    resolution=uv_res,
    uv_param_path=f'{root}/body_models/smpl_uv_decomr.npz',
    bound=(0, 1),
)
depth_renderer = dict(type='depth',
                      resolution=uv_res,
                      blend_params=dict(background_color=(0.0, 0.0, 0.0)))

convention_pred = 'lsp'

pred_kp3d = True
model = dict(
    type='PDEEstimator',
    pred_kp3d=pred_kp3d,
    use_d_weight=False,
    test_joints3d=True,
    convention_pred=convention_pred,
    mesh_sampler=dict(filename=f'{root}/mmhuman_data/mesh_downsampling.npz'),
    uv_renderer=uv_renderer,
    depth_renderer=depth_renderer,
    visualizer=dict(type='SmplVisualizer',
                    body_model=body_model_train,
                    num_per_batch=10,
                    num_batch=-1,
                    full_image=True,
                    demo_root='demo_pde'),
    backbone=dict(
        type='ResNet',
        depth=50,
        out_indices=[0, 1, 2, 3],
        norm_eval=False,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        init_cfg=dict(
            type='Pretrained',
            map_location='cpu',
            checkpoint=f'{root}/mmhuman_data/resnet50_coco_pose.pth'),
    ),
    neck=dict(
        type='DenseFPN',
        channel_dim=2,
        norm_type='BN',
        channel_list=[3, 64, 256, 512, 1024, 2048],
    ),
    head_keys=['warped_d_img', 'vertex_uv'],
    verts_head=dict(
        type='PDEHead',
        convention=convention_pred,
        pred_kp3d=pred_kp3d,
        in_channels=256,
        cam_feat_dim=2048,
        nhdim=512,
        nhdim_cam=256,
        znet_config=dict(
            number_of_embed=1,  # 24 + 10
            embed_dim=256,
            nhead=4,
            dim_feedforward=1024,
            numlayers=2,
        ),
    ),
    iuvd_head=dict(
        type='IUVDFHead2',
        norm_type='BN',
        in_channels=256,
    ),
    body_model_train=body_model_train,
    body_model_test=body_model_test,
    convention=convention,
    loss_keypoints3d=dict(type='L1Loss', loss_weight=100),
    # loss_joints3d=dict(type='L1Loss', loss_weight=100),
    loss_keypoints2d_prompt=dict(type='L1Loss', loss_weight=10),
    loss_keypoints2d_hmr=dict(type='L1Loss', loss_weight=5),
    loss_vertex=dict(type='L1Loss', loss_weight=33),
    loss_vertex_sub1=dict(type='L1Loss', loss_weight=33),
    loss_vertex_sub2=dict(type='L1Loss', loss_weight=33),
    ###
    full_uvd=True,
    loss_transl_z=dict(type='SmoothL1Loss', loss_weight=1),
    loss_iuv=dict(type='MSELoss', loss_weight=30),
    loss_image_grad_u=dict(type='ImageGradMSELoss',
                           direction='both',
                           stride=1,
                           loss_weight=10),
    loss_image_grad_v=dict(type='ImageGradMSELoss',
                           direction='both',
                           stride=1,
                           loss_weight=10),
    loss_distortion_img=dict(type='MSELoss', loss_weight=10),
    loss_wrapped_distortion=dict(type='MSELoss', loss_weight=10),
    loss_camera=dict(type='CameraPriorLoss', loss_weight=0.1),
)

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
    train=pw3d_train_transl,
    test=test_dict,
)
