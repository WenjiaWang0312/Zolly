root = '/home/wenjiawang/datasets/'

convention = 'smpl_54'
convention_test = 'h36m'  # 3dpw
# convention_test = 'smpl'  # pdhuman-syn-test, pdhuman-mtp
img_res = 224

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

checkpoint_config = dict(interval=20, )

data_keys = [
    'has_smpl', 'has_uvd', 'has_transl', 'has_focal_length', 'has_keypoints2d',
    'has_keypoints3d', 'smpl_body_pose', 'smpl_global_orient', 'smpl_betas',
    'smpl_transl', 'keypoints2d', 'keypoints3d', 'sample_idx', 'has_K', 'K',
    'is_flipped', 'origin_keypoints2d', 'smpl_origin_orient', 'ori_shape',
    'center', 'scale', 'bbox_info', 'orig_focal_length', 'crop_trans',
    'inv_trans', 'trans', 'img_h', 'img_w', 'distortion_max', 'is_distorted'
]

data_keys_spec = [
    'has_smpl', 'has_uvd', 'has_transl', 'has_focal_length', 'has_keypoints2d',
    'has_keypoints3d', 'smpl_body_pose', 'smpl_global_orient', 'smpl_betas',
    'smpl_transl', 'keypoints2d', 'keypoints3d', 'sample_idx', 'is_flipped',
    'origin_keypoints2d', 'ori_shape', 'center', 'scale', 'bbox_info',
    'orig_focal_length', 'crop_trans', 'full_img'
]

data_keys_test = [
    'has_smpl', 'has_uvd', 'has_transl', 'has_focal_length', 'smpl_body_pose',
    'has_keypoints3d', 'smpl_global_orient', 'smpl_betas', 'smpl_transl',
    'keypoints2d', 'keypoints3d', 'sample_idx', 'has_K', 'K',
    'origin_keypoints2d', 'ori_shape', 'center', 'scale', 'bbox_info',
    'orig_focal_length', 'img_h', 'img_w'
]

data_keys_inference = [
    'ori_shape', 'center', 'scale', 'bbox_info', 'orig_focal_length'
]

data_keys_adv = [
    'smpl_body_pose', 'smpl_global_orient', 'smpl_betas', 'smpl_transl'
]
train_pipeline_adv = [dict(type='Collect', keys=data_keys_adv, meta_keys=[])]

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
uv_res = 56
train_pipeline_norot = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomChannelNoise', noise_factor=0.4, img_fields=['img']),
    dict(type='RandomHorizontalFlip',
         flip_prob=0.5,
         convention=convention,
         img_fields=['img']),
    dict(type='GetRandomScaleRotation',
         rot_factor=0,
         scale_factor=0.25,
         rot_prob=0),
    dict(type='GetBboxInfo'),
    dict(type='MeshAffine',
         img_res=dict(img=224),
         require_origin_kp2d=True,
         img_fields=['img']),
    dict(type='Normalize', img_fields=['img'], **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect',
         keys=['img', *data_keys],
         meta_keys=['image_path', 'center', 'scale', 'rotation'])
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomChannelNoise', noise_factor=0.4, img_fields=['img']),
    dict(type='RandomHorizontalFlip',
         flip_prob=0.5,
         convention=convention,
         img_fields=['img']),
    dict(type='GetRandomScaleRotation',
         rot_factor=30,
         scale_factor=0.25,
         rot_prob=0.5),
    dict(type='GetBboxInfo'),
    dict(type='MeshAffine',
         img_res=dict(img=224),
         require_origin_kp2d=True,
         img_fields=['img']),
    dict(type='Normalize', img_fields=['img'], **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect',
         keys=['img', *data_keys],
         meta_keys=['image_path', 'center', 'scale', 'rotation'])
]

train_pipeline_spec = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomChannelNoise', noise_factor=0.4, img_fields=['img']),
    dict(type='RandomHorizontalFlip',
         flip_prob=0.5,
         convention=convention,
         img_fields=['img']),
    dict(type='GetRandomScaleRotation',
         rot_factor=30,
         scale_factor=0.25,
         rot_prob=0.5),
    dict(type='GetBboxInfo'),
    dict(type='MeshAffine',
         img_res=dict(img=img_res, full_img=600),
         require_origin_kp2d=True,
         img_fields=['img', 'full_img']),
    dict(type='Normalize', img_fields=['img', 'full_img'], **img_norm_cfg),
    dict(type='ImageToTensor', keys=[
        'img',
        'full_img',
    ]),
    dict(type='ToTensor', keys=data_keys_spec),
    dict(type='Collect',
         keys=['img', 'full_img', *data_keys_spec],
         meta_keys=['image_path', 'center', 'scale', 'rotation'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='GetBboxInfo'),
    dict(type='MeshAffine', img_res=img_res),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys_test),
    dict(type='Collect',
         keys=['img', *data_keys_test],
         meta_keys=['image_path', 'center', 'scale', 'rotation'])
]
test_pipeline_spec = [
    dict(type='LoadImageFromFile'),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='GetBboxInfo'),
    dict(type='MeshAffine',
         img_res=dict(img=img_res, full_img=600),
         require_origin_kp2d=True,
         img_fields=['img', 'full_img']),
    dict(type='Normalize', img_fields=['img', 'full_img'], **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img', 'full_img']),
    dict(type='ToTensor', keys=data_keys_test),
    dict(type='Collect',
         keys=['img', 'full_img', *data_keys_test],
         meta_keys=['image_path', 'center', 'scale', 'rotation'])
]

inference_pipeline_spec = [
    dict(type='GetRandomScaleRotation',
         rot_factor=0,
         scale_factor=0,
         rot_prob=0),
    dict(type='GetBboxInfo'),
    dict(type='MeshAffine',
         img_res=dict(img=img_res, full_img=600),
         require_origin_kp2d=True,
         img_fields=['img', 'full_img']),
    dict(type='Normalize', img_fields=['img', 'full_img'], **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img', 'full_img']),
    dict(type='ToTensor', keys=data_keys_inference),
    dict(type='Collect',
         keys=['img', 'full_img', 'sample_idx', *data_keys_inference],
         meta_keys=['image_path', 'center', 'scale', 'rotation'])
]

inference_pipeline = [
    dict(type='GetRandomScaleRotation',
         rot_factor=0,
         scale_factor=0,
         rot_prob=0),
    dict(type='GetBboxInfo'),
    dict(type='MeshAffine', img_res=img_res),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys_inference),
    dict(type='Collect',
         keys=['img', 'sample_idx', *data_keys_inference],
         meta_keys=['image_path', 'center', 'scale', 'rotation'])
]

cache_files = {
    'h36m': f'{root}/cache/h36m_train_{convention}.npz',
    'h36m_mosh': f'{root}/cache/h36m_mosh_train_{convention}.npz',
    'h36m_transl': f'{root}/cache/h36m_mosh_train_transl_{convention}.npz',
    'mpi_inf_3dhp': f'{root}/cache/mpi_inf_3dhp_train_{convention}.npz',
    'lsp': f'{root}/cache/lsp_train_{convention}.npz',
    'lspet_eft': f'{root}/cache/lspet_eft_{convention}.npz',
    'mpii': f'{root}/cache/mpii_train_{convention}.npz',
    'mpii_cliff': f'{root}/cache/mpii_cliff_{convention}.npz',
    'muco': f'{root}/cache/muco_{convention}.npz',
    'coco2014': f'{root}/cache/coco_2014_train_{convention}.npz',
    'coco_cliff': f'{root}/cache/coco_cliff_train_{convention}.npz',
    'coco2017': f'{root}/cache/coco_2017_train_{convention}.npz',
    'spec_train': f'{root}/cache/spec_train_{convention}.npz',
    'agora_train': f'{root}/cache/agora_train_{convention}.npz',
    'agora_transl': f'{root}/cache/agora_train_transl_{convention}.npz',
    'agora_val_transl': f'{root}/cache/agora_val_transl_{convention}.npz',
    'pw3d_train': f'{root}/cache/pw3d_train_{convention}.npz',
    'pw3d_transl': f'{root}/cache/pw3d_train_transl_{convention}.npz',
    'pdhuman_train': f'{root}/cache/pdhuman_train_{convention}.npz',
    'pdhuman_train2': f'{root}/cache/pdhuman_train2_{convention}.npz',
    'humman_train': f'{root}/cache/humman_train_{convention}.npz',
    'humman_train2': f'{root}/cache/humman_train2_{convention}.npz',
    'spec_mtp': f'{root}/cache/spec_mtp_{convention}.npz',
}

body_model_3dpw = dict(
    type='GenderedSMPL',
    keypoint_src='h36m',
    keypoint_dst='h36m',
    model_path=f'{root}/body_models/smpl',
    joints_regressor=f'{root}/body_models/J_regressor_h36m.npy')

body_model_test = dict(
    type='SMPL',
    keypoint_src='h36m',
    keypoint_dst='h36m',
    model_path=f'{root}/body_models/smpl',
    joints_regressor=f'{root}/body_models/J_regressor_h36m.npy')

body_model_train = dict(
    type='SMPL',
    keypoint_src='smpl_54',
    keypoint_dst=convention,
    model_path=f'{root}/body_models/smpl',
    keypoint_approximate=True,
    extra_joints_regressor=f'{root}/body_models/J_regressor_extra.npy')

visualizer = dict(type='SmplVisualizer',
                  body_model=body_model_train,
                  num_per_batch=-1,
                  num_batch=-1,
                  demo_root='/mnt/lustre/wangwenjia/programs/demo')

pdhuman_train2 = dict(type='HumanImageDataset',
                      data_prefix=f'{root}/mmhuman_data/',
                      convention=convention,
                      ann_file='pdhuman_train2.npz',
                      dataset_name='pdhuman',
                      is_distorted=True,
                      cache_data_path=cache_files['pdhuman_train2'],
                      pipeline=train_pipeline_norot)

pdhuman_train = dict(type='HumanImageDataset',
                     data_prefix=f'{root}/mmhuman_data/',
                     convention=convention,
                     ann_file='pdhuman_train.npz',
                     dataset_name='pdhuman',
                     is_distorted=True,
                     cache_data_path=cache_files['pdhuman_train'],
                     pipeline=train_pipeline_norot)

h36m_mosh = dict(type='HumanImageDataset',
                 dataset_name='h36m',
                 data_prefix=f'{root}/mmhuman_data/',
                 pipeline=train_pipeline,
                 convention=convention,
                 cache_data_path=cache_files['h36m_mosh'],
                 ann_file='h36m_mosh_train.npz')

h36m_mosh_transl = dict(type='HumanImageDataset',
                        dataset_name='h36m',
                        data_prefix=f'{root}/mmhuman_data/',
                        pipeline=train_pipeline,
                        convention=convention,
                        cache_data_path=cache_files['h36m_transl'],
                        ann_file='h36m_mosh_train_transl.npz')

h36m_mosh_transl_3w = dict(type='HumanImageDataset',
                           dataset_name='h36m',
                           num_data=30000,
                           data_prefix=f'{root}/mmhuman_data/',
                           pipeline=train_pipeline,
                           convention=convention,
                           cache_data_path=cache_files['h36m_transl'],
                           ann_file='h36m_mosh_train_transl.npz')

coco_2014 = dict(type='HumanImageDataset',
                 dataset_name='coco',
                 data_prefix=f'{root}/mmhuman_data/',
                 pipeline=train_pipeline,
                 convention=convention,
                 cache_data_path=cache_files['coco2014'],
                 ann_file='eft_coco_all.npz')

coco_2017 = dict(type='HumanImageDataset',
                 dataset_name='coco',
                 data_prefix=f'{root}/mmhuman_data/',
                 pipeline=train_pipeline,
                 cache_data_path=cache_files['coco2017'],
                 convention=convention,
                 ann_file='hybriK_coco_2017_train.npz')

lspet_eft = dict(type='HumanImageDataset',
                 dataset_name='lspet',
                 data_prefix=f'{root}/mmhuman_data/',
                 pipeline=train_pipeline,
                 cache_data_path=cache_files['lspet_eft'],
                 convention=convention,
                 ann_file='eft_lspet.npz')

mpi_inf_3dhp = dict(type='HumanImageDataset',
                    convention=convention,
                    dataset_name='mpi_inf_3dhp',
                    data_prefix=f'{root}/mmhuman_data/',
                    pipeline=train_pipeline,
                    cache_data_path=cache_files['mpi_inf_3dhp'],
                    ann_file='mpi_inf_3dhp_train.npz')

pw3d_train = dict(type='HumanImageDataset',
                  convention=convention,
                  dataset_name='pw3d',
                  data_prefix=f'{root}/mmhuman_data/',
                  pipeline=train_pipeline,
                  cache_data_path=cache_files['pw3d_train'],
                  ann_file='pw3d_train.npz')
pw3d_train_transl = dict(type='HumanImageDataset',
                         convention=convention,
                         dataset_name='pw3d',
                         data_prefix=f'{root}/mmhuman_data/',
                         pipeline=train_pipeline,
                         cache_data_path=cache_files['pw3d_transl'],
                         ann_file='pw3d_train_transl.npz')

spec_train = dict(type='HumanImageDataset',
                  convention=convention,
                  dataset_name='spec',
                  data_prefix=f'{root}/mmhuman_data/',
                  cache_data_path=cache_files['spec_train'],
                  pipeline=train_pipeline,
                  ann_file='spec_train_smpl.npz')

agora_train = dict(type='HumanImageDataset',
                   convention=convention,
                   dataset_name='agora',
                   data_prefix=f'{root}/mmhuman_data/',
                   cache_data_path=cache_files['agora_train'],
                   pipeline=train_pipeline,
                   ann_file='agora_train_smpl.npz')

agora_train_transl = dict(type='HumanImageDataset',
                          convention=convention,
                          dataset_name='agora',
                          data_prefix=f'{root}/mmhuman_data/',
                          cache_data_path=cache_files['agora_transl'],
                          pipeline=train_pipeline,
                          ann_file='agora_train_smpl_transl.npz')

agora_validation_transl = dict(type='HumanImageDataset',
                               convention=convention,
                               dataset_name='agora',
                               data_prefix=f'{root}/mmhuman_data/',
                               cache_data_path=cache_files['agora_val_transl'],
                               pipeline=train_pipeline,
                               ann_file='agora_validation_smpl_transl.npz')

muco = dict(type='HumanImageDataset',
            convention=convention,
            dataset_name='muco',
            data_prefix=f'{root}/mmhuman_data/',
            pipeline=train_pipeline,
            cache_data_path=cache_files['muco'],
            ann_file='muco3dhp_train.npz')

mpii = dict(type='HumanImageDataset',
            convention=convention,
            dataset_name='mpii',
            data_prefix=f'{root}/mmhuman_data/',
            pipeline=train_pipeline,
            cache_data_path=cache_files['mpii'],
            ann_file='mpii_train.npz')

mpi_inf_3dhp = dict(type='HumanImageDataset',
                    convention=convention,
                    dataset_name='mpi_inf_3dhp',
                    data_prefix=f'{root}/mmhuman_data/',
                    pipeline=train_pipeline,
                    cache_data_path=cache_files['mpi_inf_3dhp'],
                    ann_file='mpi_inf_3dhp_train.npz')

coco_cliff = dict(type='HumanImageDataset',
                  dataset_name='coco',
                  data_prefix=f'{root}/mmhuman_data/',
                  pipeline=train_pipeline,
                  convention=convention,
                  cache_data_path=cache_files['coco_cliff'],
                  ann_file='cliff_coco_train.npz')

mpii_cliff = dict(type='HumanImageDataset',
                  convention=convention,
                  dataset_name='mpii',
                  data_prefix=f'{root}/mmhuman_data/',
                  pipeline=train_pipeline,
                  cache_data_path=cache_files['mpii_cliff'],
                  ann_file='cliff_mpii_train.npz')

pdhuman_test = dict(type='HumanImageDataset',
                    data_prefix=f'{root}/mmhuman_data/',
                    convention=convention_test,
                    ann_file='pdhuman_test.npz',
                    dataset_name='pdhuman',
                    test_mode=True,
                    is_distorted=True,
                    body_model=body_model_test,
                    pipeline=test_pipeline)

# pdhuman_test2 = dict(type='HumanImageDataset',
#                      data_prefix=f'{root}/mmhuman_data/',
#                      convention=convention_test,
#                      ann_file='pdhuman_test2.npz',
#                      dataset_name='pdhuman',
#                      test_mode=True,
#                      is_distorted=True,
#                      body_model=body_model_test,
#                      pipeline=test_pipeline)

h36m_test = dict(type='HumanImageDataset',
                 dataset_name='h36m',
                 data_prefix=f'{root}/mmhuman_data/',
                 pipeline=test_pipeline,
                 body_model=body_model_test,
                 test_mode=True,
                 is_distorted=False,
                 convention=convention_test,
                 ann_file='h36m_valid_protocol2.npz')

h36m_test_joints = dict(type='HumanImageDataset',
                        dataset_name='h36m',
                        data_prefix=f'{root}/mmhuman_data/',
                        pipeline=test_pipeline,
                        body_model=body_model_test,
                        test_mode=True,
                        is_distorted=False,
                        convention=convention_test,
                        ann_file='h36m_valid_smpl.npz')

h36m_test_joints_orig = dict(type='HumanImageDataset',
                             dataset_name='h36m',
                             data_prefix=f'{root}/mmhuman_data/',
                             pipeline=test_pipeline,
                             body_model=body_model_test,
                             test_mode=True,
                             is_distorted=False,
                             convention=convention_test,
                             ann_file='h36m_valid_protocol2_with_smpl.npz')

pdhuman_test_p1 = pdhuman_test.copy()
pdhuman_test_p1['ann_file'] = 'pdhuman_test_p1.npz'
pdhuman_test_p2 = pdhuman_test.copy()
pdhuman_test_p2['ann_file'] = 'pdhuman_test_p2.npz'
pdhuman_test_p3 = pdhuman_test.copy()
pdhuman_test_p3['ann_file'] = 'pdhuman_test_p3.npz'
pdhuman_test_p4 = pdhuman_test.copy()
pdhuman_test_p4['ann_file'] = 'pdhuman_test_p4.npz'
pdhuman_test_p5 = pdhuman_test.copy()
pdhuman_test_p5['ann_file'] = 'pdhuman_test_p5.npz'

spec_mtp = dict(
    type='HumanImageDataset',
    data_prefix=f'{root}/mmhuman_data/',
    convention=convention,
    ann_file='spec_mtp.npz',
    dataset_name='spec_mtp',
    test_mode=True,
    body_model=body_model_test,
    pipeline=test_pipeline,
)

spec_mtp_train = dict(
    type='HumanImageDataset',
    data_prefix=f'{root}/mmhuman_data/',
    convention=convention,
    ann_file='spec_mtp.npz',
    dataset_name='spec_mtp',
    body_model=body_model_train,
    pipeline=train_pipeline,
    cache_data_path=cache_files['spec_mtp'],
)

spec_mtp_p1 = spec_mtp.copy()
spec_mtp_p1['ann_file'] = 'spec_mtp_p1.npz'
spec_mtp_p2 = spec_mtp.copy()
spec_mtp_p2['ann_file'] = 'spec_mtp_p2.npz'
spec_mtp_p3 = spec_mtp.copy()
spec_mtp_p3['ann_file'] = 'spec_mtp_p3.npz'

pw3d_test = dict(type='HumanImageDataset',
                 body_model=body_model_3dpw,
                 dataset_name='pw3d',
                 test_mode=True,
                 data_prefix=f'{root}/mmhuman_data/',
                 pipeline=test_pipeline,
                 ann_file='pw3d_test.npz')

# pw3d_test_pc = dict(type='HumanImageDataset',
#                     body_model=body_model_3dpw,
#                     dataset_name='pw3d',
#                     test_mode=True,
#                     data_prefix=f'{root}/mmhuman_data/',
#                     pipeline=test_pipeline,
#                     ann_file='pw3d_all_pc.npz')

# pw3d_test_oc = dict(type='HumanImageDataset',
#                     body_model=body_model_3dpw,
#                     dataset_name='pw3d',
#                     test_mode=True,
#                     data_prefix=f'{root}/mmhuman_data/',
#                     pipeline=test_pipeline,
#                     ann_file='pw3d_all_oc.npz')

pw3d_test_transl = dict(type='HumanImageDataset',
                        body_model=body_model_3dpw,
                        dataset_name='pw3d',
                        data_prefix=f'{root}/mmhuman_data/',
                        pipeline=test_pipeline,
                        test_mode=True,
                        ann_file='pw3d_test_transl.npz')

pw3d_test_transl_p1 = pw3d_test_transl
pw3d_test_transl_p2 = dict(type='HumanImageDataset',
                           body_model=body_model_3dpw,
                           dataset_name='pw3d',
                           data_prefix=f'{root}/mmhuman_data/',
                           pipeline=test_pipeline,
                           test_mode=True,
                           ann_file='pw3d_test_transl_p2.npz')
pw3d_test_transl_p3 = dict(type='HumanImageDataset',
                           body_model=body_model_3dpw,
                           dataset_name='pw3d',
                           data_prefix=f'{root}/mmhuman_data/',
                           pipeline=test_pipeline,
                           test_mode=True,
                           ann_file='pw3d_test_transl_p3.npz')
pw3d_test_transl_p4 = dict(type='HumanImageDataset',
                           body_model=body_model_3dpw,
                           dataset_name='pw3d',
                           data_prefix=f'{root}/mmhuman_data/',
                           pipeline=test_pipeline,
                           test_mode=True,
                           ann_file='pw3d_test_transl_p4.npz')

humman_train = dict(type='HumanImageDataset',
                    body_model=body_model_train,
                    convention=convention,
                    dataset_name='humman',
                    is_distorted=True,
                    data_prefix=f'{root}/mmhuman_data/',
                    pipeline=train_pipeline_norot,
                    cache_data_path=cache_files['humman_train'],
                    ann_file='humman_train.npz')

humman_train2 = dict(type='HumanImageDataset',
                     body_model=body_model_train,
                     convention=convention,
                     dataset_name='humman',
                     is_distorted=True,
                     data_prefix=f'{root}/mmhuman_data/',
                     pipeline=train_pipeline_norot,
                     cache_data_path=cache_files['humman_train2'],
                     ann_file='humman_train_2.npz')

humman_test = dict(type='HumanImageDataset',
                   body_model=body_model_test,
                   dataset_name='humman',
                   is_distorted=True,
                   test_mode=True,
                   data_prefix=f'{root}/mmhuman_data/',
                   pipeline=test_pipeline,
                   ann_file='humman_test.npz')

humman_test_p1 = dict(type='HumanImageDataset',
                      body_model=body_model_test,
                      dataset_name='humman',
                      is_distorted=True,
                      test_mode=True,
                      data_prefix=f'{root}/mmhuman_data/',
                      pipeline=test_pipeline,
                      ann_file='humman_test_p1.npz')

humman_test_p2 = dict(type='HumanImageDataset',
                      body_model=body_model_test,
                      dataset_name='humman',
                      is_distorted=True,
                      test_mode=True,
                      data_prefix=f'{root}/mmhuman_data/',
                      pipeline=test_pipeline,
                      ann_file='humman_test_p2.npz')

humman_test_p3 = dict(type='HumanImageDataset',
                      body_model=body_model_test,
                      dataset_name='humman',
                      is_distorted=True,
                      test_mode=True,
                      data_prefix=f'{root}/mmhuman_data/',
                      pipeline=test_pipeline,
                      ann_file='humman_test_p3.npz')

cmu_mosh = dict(type='MeshDataset',
                dataset_name='cmu_mosh',
                data_prefix=f'{root}/mmhuman_data/',
                pipeline=train_pipeline_adv,
                ann_file='cmu_mosh.npz')

demo_real = dict(type='DemoDataset',
                 body_model=body_model_3dpw,
                 ext='.jpg',
                 dataset_name='demo',
                 data_prefix=f'{root}/demo_jpg/',
                 pipeline=inference_pipeline)

demo_tuch = dict(type='DemoDataset',
                 body_model=body_model_3dpw,
                 dataset_name='tuch_demo',
                 data_prefix=f'{root}/mtp/images/3dcpmocap/',
                 pipeline=inference_pipeline)

test_dict = dict(
    demo_real=demo_real,
    demo_tuch=demo_tuch,
    pw3d_transl=pw3d_test_transl,
    pw3d=pw3d_test,
    humman=humman_test,
    h36m_test=h36m_test,
    h36m_test_joints=h36m_test_joints,
    h36m_test_joints_orig=h36m_test_joints_orig,
    pw3d_transl_p1=pw3d_test_transl_p1,  # 1.00
    pw3d_transl_p2=pw3d_test_transl_p2,  # 1.08
    pw3d_transl_p3=pw3d_test_transl_p3,  # 1.16
    pw3d_transl_p4=pw3d_test_transl_p4,  # 1.24
    humman_p1=humman_test_p1,  # 1.0
    humman_p2=humman_test_p2,  # 1.4
    humman_p3=humman_test_p3,  # 1.8
    pdhuman_p1=pdhuman_test_p1,  # 1.4
    pdhuman_p2=pdhuman_test_p2,  # 1.8
    pdhuman_p3=pdhuman_test_p3,  # 2.2
    pdhuman_p4=pdhuman_test_p4,  # 2.6
    pdhuman_p5=pdhuman_test_p5,  # 3.0
    spec_mtp_p1=spec_mtp_p1,  # 1.0
    spec_mtp_p2=spec_mtp_p2,  # 1.4
    spec_mtp_p3=spec_mtp_p3)  # 1.8

train_dict_pdhuman = dict(type='MixedDataset',
                          configs=[
                              pdhuman_train, h36m_mosh_transl, mpii_cliff,
                              coco_cliff, lspet_eft
                          ],
                          partition=[0.15, 0.45, 0.1, 0.2, 0.1])

train_dict_real = dict(
    type='MixedDataset',
    configs=[pdhuman_train, humman_train, lspet_eft, spec_mtp_train],
    partition=[0.4, 0.1, 0.2, 0.3])

train_dict_pdhuman2 = dict(type='MixedDataset',
                           configs=[
                               pdhuman_train2, h36m_mosh_transl, mpii_cliff,
                               coco_cliff, lspet_eft
                           ],
                           partition=[0.1, 0.5, 0.1, 0.2, 0.1])

train_dict_joint = dict(type='MixedDataset',
                        configs=[
                            pdhuman_train, pw3d_train_transl, h36m_mosh_transl,
                            mpii_cliff, coco_cliff, lspet_eft
                        ],
                        partition=[0.15, 0.1, 0.35, 0.1, 0.2, 0.1])

train_dict_all = dict(type='MixedDataset',
                      configs=[
                          pdhuman_train, humman_train, pw3d_train_transl,
                          h36m_mosh_transl, mpii_cliff, coco_cliff, lspet_eft
                      ],
                      partition=[0.15, 0.1, 0.1, 0.3, 0.05, 0.2, 0.1])

train_dict_zollyp = dict(type='MixedDataset',
                         configs=[
                             pdhuman_train, humman_train, agora_train_transl,
                             h36m_mosh_transl, coco_cliff, lspet_eft
                         ],
                         partition=[0.2, 0.15, 0.15, 0.3, 0.1, 0.1])

train_dict_zollyp_d = dict(type='MixedDataset',
                           configs=[
                               pdhuman_train, humman_train, agora_train_transl,
                               h36m_mosh_transl, coco_cliff, lspet_eft,
                               mpii_cliff
                           ],
                           partition=[0.25, 0.2, 0.1, 0.2, 0.15, 0.05, 0.05])

train_dict_zollyp2 = dict(type='MixedDataset',
                          configs=[
                              pdhuman_train, humman_train, agora_train_transl,
                              h36m_mosh_transl, coco_cliff, lspet_eft,
                              mpii_cliff
                          ],
                          partition=[0.2, 0.15, 0.1, 0.3, 0.15, 0.05, 0.05])

train_dict_humman = dict(type='MixedDataset',
                         configs=[
                             pdhuman_train, humman_train, h36m_mosh_transl,
                             mpii_cliff, coco_cliff, lspet_eft
                         ],
                         partition=[0.2, 0.2, 0.3, 0.1, 0.15, 0.05])
# mpii: 16383, lspet: 6829, h36m: 312188 coco: 28344

train_dict_spec = dict(type='MixedDataset',
                       configs=[
                           pdhuman_train, humman_train2, h36m_mosh_transl,
                           mpii_cliff, coco_cliff, lspet_eft, spec_train
                       ],
                       partition=[0.2, 0.1, 0.3, 0.05, 0.15, 0.05, 0.15])

train_dict_humman2 = dict(type='MixedDataset',
                          configs=[
                              pdhuman_train2, humman_train, h36m_mosh_transl,
                              mpii_cliff, coco_cliff, lspet_eft
                          ],
                          partition=[0.2, 0.15, 0.35, 0.1, 0.1, 0.05])

train_dict_pw3d = dict(type='MixedDataset',
                       configs=[
                           h36m_mosh_transl, pw3d_train_transl, mpii_cliff,
                           coco_cliff, lspet_eft, mpi_inf_3dhp
                       ],
                       partition=[0.4, 0.2, 0.05, 0.15, 0.05, 0.15])

train_dict_no_d = dict(type='MixedDataset',
                       configs=[
                           h36m_mosh_transl, mpii_cliff, coco_cliff, lspet_eft,
                           mpi_inf_3dhp
                       ],
                       partition=[0.45, 0.1, 0.2, 0.05, 0.2])

train_dict_h36m = dict(type='MixedDataset',
                       configs=[h36m_mosh_transl, mpi_inf_3dhp, humman_train],
                       partition=[0.8, 0.1, 0.1])

train_dict_pw3d_ft = dict(
    type='MixedDataset',
    configs=[h36m_mosh_transl_3w, pw3d_train_transl, humman_train, coco_cliff],
    partition=[0.2, 0.6, 0.1, 0.1])
