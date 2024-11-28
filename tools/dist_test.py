import argparse
import copy
import json
import os
import shutil
import sys

# try:
#     import apex
# except:
#     print("No APEX!")
import numpy as np
import torch
import yaml
from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
from ipdb import set_trace
from det3d.torchie.trainer import get_dist_info, load_checkpoint
from det3d.torchie.trainer.utils import all_gather, synchronize
from torch.nn.parallel import DistributedDataParallel
import pickle 
import time 
import wandb

def save_pred(pred, root):
    pklfile_prefix = os.path.join(root, "prediction")
    result_path = f"{pklfile_prefix}.pkl"
    with open(result_path, "wb") as f:
        pickle.dump(pred, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Eval a detector")
    parser.add_argument("config", help="config file path")
    parser.add_argument("--work_dir", required=True, help="the dir to save logs and models")
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from"
    )
    parser.add_argument(
        "--txt_result",
        type=bool,
        default=False,
        help="whether to save results to standard KITTI format of txt type",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--speed-test", action="store_true")
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--out-suffix", type=str, default='1')
    parser.add_argument("--testset", action="store_true")
    parser.add_argument("--load-preds", type=str, default=None)
    parser.add_argument("--train", default=False, action="store_true")
    parser.add_argument("--filter_ad", default=False, action="store_true")

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main(args):

    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(0)

    cfg = Config.fromfile(args.config)
    cfg.local_rank = int(os.environ["LOCAL_RANK"])

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    
    # dump config
    # if not os.path.exists(osp.join(cfg.work_dir)):
    #     os.makedirs(osp.join(cfg.work_dir), exist_ok=True)
    # shutil.copy('/workspace/CenterPoint/configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py', osp.join(cfg.work_dir, 'test_' + osp.basename(args.config)))

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

        cfg.gpus = torch.distributed.get_world_size()
    else:
        cfg.gpus = args.gpus

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info("Distributed testing: {}".format(distributed))
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    if args.testset:
        print("Use Test Config")
        dataset = build_dataset(cfg.data.test)
    else:
        print("Use Val Config")
        dataset = build_dataset(cfg.data.val)

    if args.load_preds:
        predictions = []
        result_dict, _ = dataset.evaluation(
            copy.deepcopy(predictions), output_dir=args.work_dir, testset=args.testset, train=args.train, res_path=args.load_preds, filter_ad=args.filter_ad)
        if result_dict is not None:
            for k, v in result_dict["results"].items():
                print(f"Evaluation {k}: {v}")
        return

    data_loader = build_dataloader(
        dataset,
        batch_size=cfg.data.samples_per_gpu if not args.speed_test else 1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=8,
    )

    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

    # put model on gpus
    if distributed:
        # model = apex.parallel.convert_syncbn_model(model)
        model = DistributedDataParallel(
            model.cuda(cfg.local_rank),
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    else:
        # model = fuse_bn_recursively(model)
        model = model.cuda()

    model.eval()
    mode = "val"

    logger.info(f"work dir: {args.work_dir}")
    if cfg.local_rank == 0:
        prog_bar = torchie.ProgressBar(len(data_loader.dataset) // cfg.gpus)

    detections = {}
    cpu_device = torch.device("cpu")

    start = time.time()

    start = int(len(dataset) / 3)
    end = int(len(dataset) * 2 /3)

    time_start = 0 
    time_end = 0 

    for i, data_batch in enumerate(data_loader):
        if i == start:
            torch.cuda.synchronize()
            time_start = time.time()

        if i == end:
            torch.cuda.synchronize()
            time_end = time.time()

        with torch.no_grad():
            outputs = batch_processor(
                model, data_batch, train_mode=False, local_rank=cfg.local_rank,
            )
        for output in outputs:
            token = output["metadata"]["token"]
            for k, v in output.items():
                if k not in [
                    "metadata",
                ]:
                    output[k] = v.to(cpu_device)
            detections.update(
                {token: output,}
            )
            if cfg.local_rank == 0:
                prog_bar.update()

    synchronize()

    all_predictions = all_gather(detections)

    print("\n Total time per frame: ", (time_end -  time_start) / (end - start))

    if cfg.local_rank != 0:
        return

    predictions = {}
    for p in all_predictions:
        predictions.update(p)

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    save_pred(predictions, args.work_dir)

    result_dict, _ = dataset.evaluation(copy.deepcopy(predictions), output_dir=args.work_dir,
                                        testset=args.testset, train=args.train,
                                        filter_ad=args.filter_ad)

    if result_dict is not None:
        for k, v in result_dict["results"].items():
            print(f"Evaluation {k}: {v}")

    if args.txt_result:
        assert False, "No longer support kitti"

if __name__ == "__main__":
    # main()
    from ipdb import launch_ipdb_on_exception, set_trace
    with launch_ipdb_on_exception():
        # wandb.init("cp_5seed95pseudo") # project=cfg.project_name)
        args = parse_args()

        # ch = './work_dirs/5_nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/epoch_1.pth'
        # wdir = args.work_dir
        # for e in [1]:
        #     args.checkpoint = ch[:-500] + f"{e}.pth"
        #     args.work_dir = wdir + f"eval_{e}"

        main(args)
