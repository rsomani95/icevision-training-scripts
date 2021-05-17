import torch
import pytorch_lightning as pl
import torch.nn as nn

# Python imports
from pprint import pprint
from enum import Enum

# Self written imports
from utils.logging import *
from backbones.mobilenetv3 import model_splitter

# Icevision, mmdet & co.
from icevision.all import *
from icevision.parsers.coco_parser import *
from icevision.models.mmdet.utils import *
from mmcv import Config, ConfigDict
from mmdet.models import build_detector

# Other
# Some !#@$ lightning requirement will not let you use ModelEmaV2
# inside the lightning module if it's stored on GPU because it's
# not actually being used in the backward pass.....
# Storing on "cpu" gives an error...
from timm.utils.model_ema import ModelEmaV2
from pytorch_lightning import loggers as pl_loggers
from torch.nn.modules.batchnorm import _BatchNorm

# ============== CONSTANTS ================ #

BATCH_SIZE = 18
HEIGHT, WIDTH = 640, 640
DEBUG = False
SINGLE_GPU = True  # Run training on one or multiple GPUs
SCHEDULE = "1x"
LR_TYPE = "constant"
FREEZE_BLOCKS = 1

assert LR_TYPE in ["constant", "differential"]
assert SCHEDULE in ["1x", "2x"]

DEFAULT_TOTAL_SAMPLES = 16  # 8 GPUs * 2 samples per GPU
DEFAULT_LR = 0.02

NUM_GPUS = 1
if not DEBUG:
    NUM_GPUS = 1 if SINGLE_GPU else 3
SAMPLES_PER_GPU = BATCH_SIZE
TOTAL_SAMPLES = SAMPLES_PER_GPU * NUM_GPUS
LR_SCALER = TOTAL_SAMPLES / DEFAULT_TOTAL_SAMPLES
LR = LR_SCALER * DEFAULT_LR  # == 0.0675 @18, 0.06 @16, 0.12 @ 32, and so on...
print(f"{'*' * 20} LEARNING RATE: {LR} {'*' * 20}")


if LR_TYPE == "constant":
    LEARNING_RATES = dict(
        stem=None,
        # blocks = [None, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2]
        blocks=([None] * FREEZE_BLOCKS) + ([1e-2] * (7 - FREEZE_BLOCKS)),
        neck=LR,
        bbox_head=LR,
        classifier_heads=None,
    )
elif LR_TYPE == "differential":
    raise RuntimeError(f"Don't use differential LRs when training from scratch")
    LEARNING_RATES = dict(
        stem=1e-6,
        blocks=[1e-5, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3],
        neck=1e-2,
        bbox_head=1e-2,
        classifier_heads=None,
    )

if SCHEDULE == "1x":
    TRAIN_NUM_EPOCHS = 12
    LR_STEP_MILESTONES = [8, 11]
elif SCHEDULE == "2x":
    TRAIN_NUM_EPOCHS = 24
    LR_STEP_MILESTONES = [16, 22]
if DEBUG:
    TRAIN_NUM_EPOCHS = 5


# ========================================= #


# ============== BUILD MODEL ================ #

class_map = icedata.coco.class_map()

model_name = "mobilenetv3_large_100_aa"
base_config_path = mmdet_configs_path / "retinanet"
config_path = base_config_path / "retinanet_r50_fpn_1x_coco.py"
cfg = Config.fromfile(config_path)

## mmdet >= 2.12 requires `ConfigDict`, not just `dict`
cfg.model.backbone = ConfigDict(
    dict(
        type=f"TIMM_{model_name}",
        pretrained=True,
        out_indices=(1, 2, 3, 4),
    )
)
cfg.model.neck.in_channels = [24, 40, 112, 960]
cfg.model.bbox_head.num_classes = len(class_map) - 1

model = build_detector(cfg.model)
# print(model)

# ============================================ #


# ============== PL LIGHTNING ADAPTER ================ #


class MobileNetV3Adapter(models.mmdet.retinanet.lightning.ModelAdapter):
    def __init__(
        self,
        model: nn.Module,
        metrics: List[Metric] = None,
        norm_eval: bool = True,
        freeze_blocks: int = FREEZE_BLOCKS,
    ):
        """
        `norm_eval`: Sets BatchNorm layers to eval mode
        `freeze_blocks`: No. of backbone blocks to be frozen. NOTE that the
                         conv stem is ALWAYS frozen. `freeze_blocks=21 will
        freeze the first two block. There are 7 blocks in mobilenetv3
        """
        super().__init__(model=model, metrics=metrics)
        self.norm_eval = norm_eval
        assert freeze_blocks <= 7
        self.freeze_blocks = freeze_blocks
        # self.model_ema = ModelEmaV2(self.model, decay=0.9999) #, device="cpu")

    def _freeze_stages(self):
        # ACRONYMS: m => model; l => layer
        m = self.model.backbone.model
        for l in [m.conv_stem, m.bn1, m.act1]:
            l.eval()
            for param in l.parameters():
                param.requires_grad = False

        for i in range(self.freeze_blocks):
            l = m.blocks[i]
            l.eval()
            for param in l.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(MobileNetV3Adapter, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for layer in self.model.backbone.model.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(layer, _BatchNorm):
                    layer.eval()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            model_splitter(
                self.model,
                LR_stem=self.LRs["stem"],
                LR_blocks=self.LRs["blocks"],
                LR_neck=self.LRs["neck"],
                LR_bbox_head=self.LRs["bbox_head"],
                LR_classifier_heads=self.LRs["classifier_heads"],
            ),  # returns a list of parameter groups
            lr=0.01,
            momentum=0.9,
            weight_decay=0.0001,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=LR_STEP_MILESTONES
        )
        return [optimizer], [scheduler]

    def optimizer_step(
        # fmt: off
        self, epoch, batch_idx, optimizer, optimizer_idx,
        optimizer_closure, on_tpu, using_native_amp, using_lbfgs,
        # fmt: on
    ):
        # warm up lr
        if self.trainer.global_step < 500:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 500.0)

            # if LR_TYPE == "differential":
            if True:
                # ensure we respect individual param groups' LR by scaling
                # each group's lr with the initial lr
                # TODO: This can be easily refactored into an `lr_scheduler`.
                # You don't need to track each group manually that way
                for pg in optimizer.param_groups:
                    if "block_idx" in pg.keys():
                        pg["lr"] = lr_scale * self.LRs["blocks"][pg["block_idx"]]
                    else:
                        pg["lr"] = lr_scale * self.LRs[pg["name"]]
            else:
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_scale * LR

        # update params, EMA
        optimizer.step(closure=optimizer_closure)
        # self.model_ema.update(self.model)

    def training_step(self, batch, batch_idx):
        data, samples = batch

        outputs = self.model.train_step(data=data, optimizer=None)
        for k, v in outputs["log_vars"].items():
            self.log(f"train/{k}", v)

        DEBUG = True
        if DEBUG and self.training:
            print(f'{"*"*30} Training Step {"*"*30}')
            print(f"Finding unused parameters...")

            for name, param in self.model.named_parameters():
                if param.grad is None:
                    print(name)
            print(f'{"*"*80}')
        DEBUG = False

        return outputs["loss"]


pl_model = MobileNetV3Adapter(
    model=model,
    metrics=[COCOMetric(metric_type=COCOMetricType.bbox)],
)

# ==================================================== #


# ============== BUILD DATASET ================ #

train_annotations_path = (
    "/home/synopsis/datasets/coco/annotations/instances_train2017.json"
)
valid_annotations_path = (
    "/home/synopsis/datasets/coco/annotations/instances_val2017.json"
)
train_data_dir = Path("/home/synopsis/datasets/coco/train2017/")
valid_data_dir = Path("/home/synopsis/datasets/coco/val2017/")

class_map = icedata.coco.class_map()

train_parser = COCOBBoxParser(train_annotations_path, train_data_dir)
train_records, _ = train_parser.parse(
    data_splitter=RandomSplitter([1.0, 0.0]),
    cache_filepath=Path.cwd() / "cache" / "CACHE--COCO-Instances-TRAIN.pkl",
)

valid_parser = COCOBBoxParser(valid_annotations_path, valid_data_dir)
valid_records, _ = valid_parser.parse(
    data_splitter=RandomSplitter([1.0, 0.0]),
    cache_filepath=Path.cwd() / "cache" / "CACHE--COCO-Instances-VAL.pkl",
)

# ============================================= #


# ============== BUILD AUGMENTATIONS ================ #

train_tfms = tfms.A.Adapter(
    [
        tfms.A.HorizontalFlip(p=0.15),
        tfms.A.VerticalFlip(p=0.15),
        tfms.A.ShiftScaleRotate(rotate_limit=25, p=0.15),
        tfms.A.Blur(p=0.15, blur_limit=(1, 3)),
        tfms.A.RandomBrightnessContrast(p=0.15),
        tfms.A.Normalize(),
        tfms.A.Resize(HEIGHT, WIDTH)
        #     tfms.A.OneOrOther(
        #         tfms.A.RandomSizedBBoxSafeCrop(HEIGHT, WIDTH),
        #         tfms.A.Resize(HEIGHT, WIDTH)
        #     )
    ]
)
valid_tfms = tfms.A.Adapter(
    [
        tfms.A.Resize(HEIGHT, WIDTH),
        tfms.A.PadIfNeeded(HEIGHT, WIDTH),
        tfms.A.Normalize(),
    ]
)

train_ds = Dataset(train_records, train_tfms)
valid_ds = Dataset(valid_records, valid_tfms)
# valid_ds = Dataset(valid_records, train_tfms)

# =================================================== #


# =================== DATALOADER ==================== #

train_dl = models.mmdet.retinanet.train_dl(
    train_ds, batch_size=BATCH_SIZE, num_workers=8, shuffle=True, pin_memory=True
)
valid_dl = models.mmdet.retinanet.valid_dl(
    valid_ds, batch_size=BATCH_SIZE * 2, num_workers=8, shuffle=False, pin_memory=True
)

# =================================================== #


# ============== DATALOADER, METRICS ================ #

loggers = [pl.loggers.TensorBoardLogger(name="coco-mnv3", save_dir="lightning_logs/")]
if not DEBUG:
    loggers.append(pl.loggers.wandb.WandbLogger(project="coco-mnv3"))

trainer = pl.Trainer(
    max_epochs=TRAIN_NUM_EPOCHS,
    gpus=-1 if NUM_GPUS > 1 else [0],
    # gpus=[0],
    benchmark=True,
    precision=16,
    accelerator="ddp" if NUM_GPUS > 1 else None,
    logger=loggers,
    callbacks=[
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.ModelCheckpoint(save_top_k=-1),
    ],
    # weights_summary="full",
    overfit_batches=1 if DEBUG else 0.0,  # 0.0 => train on full dataset
)
trainer.fit(pl_model, train_dl, valid_dl)
trainer.save_checkpoint("end.ckpt")

# =================================================== #


# ================= CONFIG LOGGING ================== #


if not DEBUG:
    wandb.config.update(
        dict(
            model="mobilenetv3-aa",
            num_epochs=TRAIN_NUM_EPOCHS,
            lr_policy=LR_TYPE,
            schedule=SCHEDULE,
            batch_size=BATCH_SIZE,
            image_size=f"{HEIGHT}x{WIDTH}",
            optimizer="SGD",
            train_tfms=[extract_tfm_string(t) for t in train_tfms.tfms_list],
        )
    )

# =================================================== #
