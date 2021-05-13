import wandb
from icevision.all import *

HEIGHT, WIDTH = 640, 640

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


def extract_tfm_string(tfm):
    info = tfm.get_dict_with_id()
    name = info["__class_fullname__"].split(".")[-1]
    prob = info["p"]
    return f"{name}__p-{prob}"


api = wandb.Api()
run = api.run("synopsis/coco-mnv3/355qclcy")
run.config["model"] = "mobilenetv3-aa"
run.config["num_epochs"] = 24
run.config["lr_policy"] = "differential"
run.config["batch_size"] = 18
run.config["image_size"] = f"{HEIGHT}x{WIDTH}"
run.config["optimizer"] = "SGD"
run.config["train_tfms"] = [extract_tfm_string(t) for t in train_tfms.tfms_list]
run.update()
