_base_ = "./cascade_rcnn_r50_fpn.py"

conv_cfg = dict(type="ConvAWSV1")
model = dict(
    type="RecursiveFeaturePyramid",
    rfp_steps=2,
    rfp_sharing=False,
    stage_with_rfp=(False, True, True, True),
    backbone=dict(
        _delete_=True,
        type="DetectoRS_ResNetV1",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        conv_cfg=conv_cfg,
        sac=dict(type="SACV1", use_deform=True),
        stage_with_sac=(False, True, True, True),
        norm_cfg=dict(type="BN", requires_grad=True),
        style="pytorch",
        gen_attention=dict(spatial_range=-1, num_heads=8, attention_type="0010", kv_stride=2),
        stage_with_gen_attention=[[], [], [0, 1, 2, 3, 4, 5], [0, 1, 2]],
    ),
)

test_cfg = dict(rcnn=dict(score_thr=0.5, nms=dict(iou_thr=0.5)))
