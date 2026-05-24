# DA-SLRA
This repository maintains the official implementation of the paper Difficulty-Aware Cross-Modal Rectification with Spectral Low-Rank Adaptation for Remote Sensing Image-Text Retrieval, currently being prepared for submission to [the Visual Computer](https://link.springer.com/journal/371).
## Installation
` pip install -r requirements.txt `
## Datasets
All experiments are conducted on RSICD and RSITMD. Download the images from Hugging Face.
- RSICD: https://doi.org/10.1109/TGRS.2017.2776321
- RSITMD: https://doi.org/10.1109/TGRS.2021.3078451

Annotation files are stored in the `.data/finetune` directory and
Modify the `image_root` parameter in YAML configuration files at `configs/*.yaml` to match your local file path.
## Training
Retrieve the pre-trained GeoRSCLIP model through [the attached link](https://huggingface.co/Zilun/GeoRSCLIP/blob/main/ckpt/RS5M_ViT-B-32_RET-2.pt), then place the downloaded checkpoint in the `open_clip/pretrain/` folder.
Execute the commands below to launch training on corresponding datasets.
- RSICD:
```
export CUDA_VISIBLE_DEVICES=0 && python -W ignore run.py --task 'itr_rsicd_vit' --dist 'gpu0' --config 'configs/Retrieval_rsicd_vit.yaml' --output_dir './checkpoints/full_rsicd_vit'
export CUDA_VISIBLE_DEVICES=0 && python -W ignore run.py --task 'itr_rsicd_geo' --dist 'gpu0' --config 'configs/Retrieval_rsicd_geo.yaml' --output_dir './checkpoints/full_rsicd_geo'
```
- RSITMD:
```
export CUDA_VISIBLE_DEVICES=0 && python -W ignore run.py --task 'itr_rsitmd_vit' --dist 'gpu0' --config 'configs/Retrieval_rsitmd_vit.yaml' --output_dir './checkpoints/full_rsitmd_vit'
export CUDA_VISIBLE_DEVICES=0 && python -W ignore run.py --task 'itr_rsitmd_geo' --dist 'gpu0' --config 'configs/Retrieval_rsitmd_geo.yaml' --output_dir './checkpoints/HARMA/full_rsitmd_geo'
```
## Evaluation
For model evaluation, you can activate the evaluation function by defining `if_evaluation: True` in config files, or directly use the `--evaluate` command line flag.
For example:
```
export CUDA_VISIBLE_DEVICES=0 && python -W ignore run.py --task itr_rsicd_vit --dist f2 --config configs/Retrieval_rsicd_vit.yaml --output_dir "/root/autodl-tmp/DA-SGLR/outputs/test" --checkpoint "/root/autodl-tmp/DA-SGLR/checkpoints/full_rsicd_vit/checkpoint_best.pth" --evaluate
```









