import pytorch_lightning as pl

from timm.utils.model_ema import ModelEmaV2
from backbones.mobilenetv3 import model_splitter
from pytorch_lightning import loggers as pl_loggers
from icevision.all import *
from icevision.models.mmdet.utils import *
from mmdet.models import build_detector
from mmcv import Config
from pprint import pprint

# ============== CONSTANTS ================ #

TRAIN_NUM_EPOCHS = 24
BATCH_SIZE = 18
DEBUG = True

# ========================================= #


# ============== BUILD MODEL ================ #

model_name = "mobilenetv3_large_100_aa"
base_config_path = mmdet_configs_path / "retinanet"
config_path = base_config_path / "retinanet_r50_fpn_1x_coco.py"
cfg = Config.fromfile(config_path)

cfg.model.backbone = dict(
    type=f"TIMM_{model_name}",
    pretrained=True,
    out_indices=(0, 1, 2, 3, 4),
)
cfg.model.neck.in_channels = [16, 24, 40, 112, 960]

model = build_detector(cfg.model)
print(model)

# optimizer = torch.optim.SGD(
#     model_splitter(model), lr=0.01, momentum=0.9, weight_decay=0.0001
# )
# print(len(optimizer.param_groups))

# ============================================ #


# ============== PL LIGHTNING ADAPTER ================ #


class MobileNetV3Adapter(models.mmdet.retinanet.lightning.ModelAdapter):
    def __init__(self, model: nn.Module, metrics: List[Metric] = None):
        super().__init__(metrics=metrics)
        self.model = model
        self.model_ema = ModelEmaV2(self.model, decay=0.9999, device="cpu")

    def configure_optimizers(self):
        self.LRs = dict(
            stem=1e-6,
            blocks=[1e-5, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3],
            neck=1e-2,
            bbox_head=1e-2,
            classifier_heads=None,
        )
        assert isinstance(self.LRs["blocks"], list)  # just do it.

        optimizer = torch.optim.SGD(
            model_splitter(
                self.model,
                LR_stem=self.LRs["stem"],
                LR_blocks=self.LRs["blocks"],
                LR_neck=self.LRs["neck"],
                LR_bbox_head=self.LRs["bbox_head"],
                LR_classifier_heads=self.LRs["classifier_heads"],
            ),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.0001,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16, 22])
        return [optimizer, scheduler]

    def optimizer_step(
        # fmt: off
        self, epoch, batch_idx, optimizer, optimizer_idx,
        optimizer_closure, on_tpu, using_native_amp, using_lbfgs,
        # fmt: on
    ):
        # warm up lr
        if self.trainer.global_step < 500:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 500.0)

            # ensure we respect individual param groups' LR by scaling
            # each group's lr with the initial lr
            for pg in optimizer.param_groups:
                if "block_idx" in pg.keys():
                    pg["lr"] = lr_scale * self.LRs["blocks"][pg["block_idx"]]
                else:
                    pg["lr"] = lr_scale * self.LRs[pg["name"]]

        # update params
        optimizer.step(closure=optimizer_closure)
        self.model_ema.update(self.model)


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

train_parser = icedata.coco.parser(train_annotations_path, train_data_dir)
train_records, _ = train_parser.parse(
    data_splitter=RandomSplitter([1.0, 0.0]),
    cache_filepath=Path.cwd() / "cache" / "CACHE--COCO-Instances-TRAIN.pkl",
)

valid_parser = icedata.coco.parser(valid_annotations_path, valid_data_dir)
valid_records, _ = valid_parser.parse(
    data_splitter=RandomSplitter([1.0, 0.0]),
    cache_filepath=Path.cwd() / "cache" / "CACHE--COCO-Instances-VAL.pkl",
)

# ============================================= #


# ============== BUILD AUGMENTATIONS ================ #

HEIGHT, WIDTH = 640, 640
BATCH_SIZE = 16

train_tfms = tfms.A.Adapter(
    [
        # *tfms.A.aug_tfms(size=img_size, pad=None),
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


tb_logger = pl_loggers.TensorBoardLogger(model_save_path, name=model_filename)
trainer = pl.Trainer(
    max_epochs=TRAIN_NUM_EPOCHS,
    gpus=-1,
    benchmark=True,
    precision=16,
    accelerator="ddp",  # turn multi-gpu acceleration
    logger=pl.loggers.TensorBoardLogger(
        name="coco-instances", save_dir="lightning_logs/"
    ),
    callbacks=[pl.callbacks.LearningRateMonitor(logging_interval="step")],
    weights_summary="full",
    overfit_batches=20 if DEBUG else 0.0,
)
trainer.fit(pl_model, train_dl, valid_dl)


# =================================================== #