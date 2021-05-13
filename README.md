# icevision-training-scripts

Currently using a custom variant of `mobilenet-v3`. To reproduce, install icevision, then
```bash
git clone https://github.com/rsomani95/pytorch-image-models
cd pytorch-image-models
git checkout aa-effnets
pip install -e .
```

There's an [ongoing PR](https://github.com/rwightman/pytorch-image-models/pull/603) for the branch being used.

Progress can be tracked here: https://wandb.ai/synopsis/coco-mnv3
