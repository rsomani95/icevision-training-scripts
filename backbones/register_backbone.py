import timm
from timm.models.mobilenetv3 import *
from typing import Optional, Collection

from mmcv import Config
from mmcv.utils import Registry, build_from_cfg

from mmdet.models.necks.fpn import FPN
from mmdet.models.detectors import RetinaNet
from mmdet.models.builder import BACKBONES
from mmdet.models.builder import DETECTORS
from mmdet.models.builder import *
from icevision.models.mmdet.utils import *
from mmdet.models import build_detector
from tqdm import tqdm

# timm.list_models(pretrained=True, filter="mobile*")
timm.create_model("mobilenetv3_large_100_aa", pretrained=True, features_only=True)


def model_dispatcher(model_name) -> type:
    """
    Utlility function that returns classes for specific models that can
    be registered in mmdet's `BACKBONES` registry

    Example:
    model_name = "mobilenetv3_large_100"
    BACKBONES.register_module(name=f"TIMM_{model}", module=model_dispatcher(model_name))
    """

    class TimmBackboneWrapper(nn.Module):
        """
        Wrapper for timm's `create_model`. Internally, `features_only` is always called,
        and you can optionally pass in `out_indices` for fine grained control over
        which feature maps are returned
        """

        def __init__(
            self,
            model_name: str = model_name,
            pretrained: bool = False,
            checkpoint_path="",
            scriptable: bool = None,
            exportable: bool = None,
            no_jit: bool = None,
            out_indices: Optional[Collection[int]] = None,
            **kwargs,
        ):
            super().__init__()
            if model_name is None:
                if not "model_name" in self.__dict__:
                    raise ValueError(f"You must provide a value for `model_name`")
                else:
                    model_name = self.__dict__["model_name"]
            kwargs.update(dict(features_only=True))
            if out_indices is not None:
                kwargs.update(dict(out_indices=out_indices))

            self.model = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                checkpoint_path=checkpoint_path,
                scriptable=scriptable,
                exportable=exportable,
                no_jit=no_jit,
                **kwargs,
            )

        def init_weights(self, pretrained=None):
            pass

        def forward(self, x):
            return self.model(x)

    return TimmBackboneWrapper


# FE_models = []
# TIMM_PRETRAINED_MODELS = timm.list_models(pretrained=True)
# pbar = tqdm(TIMM_PRETRAINED_MODELS, total=len(TIMM_PRETRAINED_MODELS))
# for model in pbar:
#     try:
#         pbar.set_description(f"{model}")
#         timm.create_model(model, features_only=True)
#         FE_models.append(model)

#         # We need a class that can be passed as the `module` arg in
#         # `BACKBONES.register_module`
#         model_cls = model_dispatcher(model)
#         BACKBONES.register_module(name=f"TIMM_{model}", module=model_cls)
#     except:
#         pass

# Path("timm_features_only_compatible_models.txt").write_text("\n".join(FE_models))