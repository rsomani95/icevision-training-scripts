from .register_backbone import model_dispatcher
from .model_splitter import *
from mmdet.models.builder import BACKBONES
from typing import Optional, List

model_name = "mobilenetv3_large_100_aa"
BACKBONES.register_module(
    name=f"TIMM_{model_name}", module=model_dispatcher(model_name)
)

# model_name = "mobilenetv3_large_100_aa"
# base_config_path = mmdet_configs_path / "retinanet"
# config_path = base_config_path / "retinanet_r50_fpn_1x_coco.py"
# cfg = Config.fromfile(config_path)
# cfg.model.backbone = dict(type="TIMM_mobilenetv3_large_100_aa", pretrained=True)


def model_splitter(
    m: "TimmBackboneWrapper",
    LR_stem: float = 1e-6,
    LR_blocks: Union[float, List[float]] = [1e-5, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3],
    LR_neck: float = 1e-2,
    LR_bbox_head: float = 1e-2,
    LR_classifier_heads: Optional[float] = None,
):
    """
    Splits a mobilenetv3 w/ RetinaNet-Head (or any one stage detector from mmdet)
    into multiple param groups:
    """
    if isinstance(LR_blocks, list):
        assert len(LR_blocks) == 7
    else:
        # Unroll same LR for all 7 blocks
        LR_blocks = [LR_blocks] * 7

    # b = backbone
    b = m.backbone.model

    # assign individual LR per block
    block_param_groups = [
        # assign `block_idx` for easy indexing in LR warmup
        dict(name=f"block_{i}", block_idx=i, lr=lr, params=params(block))
        for i, (block, lr) in enumerate(zip(b.blocks, LR_blocks))
    ]

    param_groups = [
        dict(name="stem", lr=LR_stem, params=params([b.conv_stem, b.bn1])),
        *block_param_groups,
        dict(name="neck", lr=LR_neck, params=params(m.neck)),
        dict(name="bbox_head", lr=LR_bbox_head, params=params(m.bbox_head)),
    ]
    if LR_classifier_heads is not None:
        param_groups.append(
            dict(
                name="classifier_heads",
                lr=LR_classifier_heads,
                params=params(m.classifier_heads),
            )
        )
    return param_groups
