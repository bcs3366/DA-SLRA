import os
import sys
import time
import random
import argparse
import torch
import re

from utils.hdfs_io import HADOOP_BIN, hexists, hmkdir, hcopy

def get_dist_launch(args):
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise ValueError("No GPUs are available.")

    # 单 GPU 环境下，指定要使用的 GPU
    if num_gpus == 1:
        # 强制使用 GPU 0
        # return "set CUDA_VISIBLE_DEVICES=0 && /root/miniconda3/bin/python -W ignore"
        return "set CUDA_VISIBLE_DEVICES=0 && /root/miniconda3/bin/python -W ignore"

    else:
        if args.dist.startswith('gpu'):
            num = int(args.dist[3:])
            if 0 <= num < num_gpus:
                # return "set CUDA_VISIBLE_DEVICES=0 && /root/miniconda3/bin/python -W ignore"
                return "set CUDA_VISIBLE_DEVICES={:} && /root/miniconda3/bin/python -W ignore".format(num)
            else:
                raise ValueError("Selected GPU number is not available.")
        else:
            # 如果不是特定的 'gpuX' 格式，则默认使用第一个 GPU
            # return "set CUDA_VISIBLE_DEVICES=0 && /root/miniconda3/bin/python -W ignore"
            return "set CUDA_VISIBLE_DEVICES=0 && /root/miniconda3/bin/python -W ignore"


def get_from_hdfs(file_hdfs):
    """
    compatible to HDFS path or local path
    """
    if file_hdfs.startswith('hdfs'):
        file_local = os.path.split(file_hdfs)[-1]

        if os.path.exists(file_local):
            print(f"rm existing {file_local}")
            os.system(f"rm {file_local}")

        hcopy(file_hdfs, file_local)

    else:
        file_local = file_hdfs
        assert os.path.exists(file_local)

    return file_local


def run_retrieval(args):
    dist_launch = get_dist_launch(args)

    # 直接运行脚本，不再使用 torch.distributed.launch
    os.system(f"{dist_launch} Retrieval.py --config {args.config} "
              f"--output_dir {args.output_dir} --bs {args.bs} --checkpoint {args.checkpoint} {'--evaluate' if args.evaluate else ''}")


def run(args):
    if args.task == 'itr_rsicd_vit':
        # assert os.path.exists("../X-VLM-pytorch/images/rsicd")
        args.config = 'configs/Retrieval_rsicd_vit.yaml'
        run_retrieval(args)

    elif args.task == 'itr_rsitmd_vit':
        # assert os.path.exists("../X-VLM-pytorch/images/rsitmd")
        args.config = 'configs/Retrieval_rsitmd_vit.yaml'
        run_retrieval(args)
    elif args.task == 'itr_rsitmd_geo':
        # assert os.path.exists("../X-VLM-pytorch/images/rsitmd")
        args.config = 'configs/Retrieval_rsitmd_geo.yaml'
        run_retrieval(args)
    elif args.task == 'itr_rsicd_geo':
        # assert os.path.exists("../X-VLM-pytorch/images/rsicd")
        args.config = 'configs/Retrieval_rsicd_geo.yaml'
        run_retrieval(args)

    elif args.task == 'itr_coco':
        assert os.path.exists("../X-VLM-pytorch/images/coco")
        args.config = 'configs/Retrieval_coco.yaml'
        run_retrieval(args)

    elif args.task == 'itr_nwpu':
        assert os.path.exists("../X-VLM-pytorch/images/NWPU")
        args.config = 'configs/Retrieval_nwpu.yaml'
        run_retrieval(args)
    else:
        raise NotImplementedError(f"task == {args.task}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='itr_rsitmd')
    parser.add_argument('--dist', type=str, default='gpu0', help="see func get_dist_launch for details")
    parser.add_argument('--config', default='configs/Retrieval_rsitmd_geo.yaml', type=str,
                        help="if not given, use default")
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus; "
                                                           "this option only works for fine-tuning scripts.")
    parser.add_argument('--seed', default=42, type=int)
    # parser.add_argument('--checkpoint', default='-1', type=str, help="for fine-tuning")
    parser.add_argument('--checkpoint', default='/root/autodl-tmp/DA-SLRA/checkpoints/full_rsitmd_geo/52.70/checkpoint_best.pth', type=str, help="for fine-tuning")
    parser.add_argument('--load_ckpt_from', default=' ', type=str, help="load domain pre-trained params")
    # write path: local or HDFS
    parser.add_argument('--output_dir', type=str, default='./outputs/test', help='for fine-tuning, local path; '
                                                                                 'for pre-training, local and HDFS are both allowed.')
    parser.add_argument('--evaluate', action='store_true', default=False, help="evaluation on downstream tasks")
    args = parser.parse_args()

    args.output_dir = re.sub(r'[\'\"]', '', args.output_dir)

    parent_dir = os.path.dirname(args.output_dir)
    assert hexists(parent_dir)

    hmkdir(args.output_dir)

    print("当前 args:", args)
    run_retrieval(args)

