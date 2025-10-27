import json
import os
import time

import torch
from loguru import logger
from timm.utils import AverageMeter

from _data import build_loader, get_topk, get_class_num
from _utils import (
    build_optimizer,
    build_scheduler,
    calc_learnable_params,
    EarlyStopping,
    init,
    print_in_md,
    save_checkpoint,
    seed_everything,
    validate_smart,
)
from config import get_config
from loss import PVSELoss
from network import build_model
from utils import mean_average_precision




def train_init(args):
    # setup net
    net, out_idx = build_model(args, True)

    # setup criterion
    criterion = PVSELoss(args)

    logger.info(f"number of learnable params: {calc_learnable_params(net)}")

    # setup optimizer
    # {"params": list(set(net.parameters()).difference(set(net.fc.parameters()))), "lr": args.lr},
    # {"params": net.fc.parameters(), "lr": args.lr_fc},
    optimizer = build_optimizer(args.optimizer, net.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.scheduler == "reduce":
        scheduler = build_scheduler(args.scheduler, optimizer, factor=0.5, min_lr=1e-10)
    elif args.scheduler == "cosine":
        scheduler = build_scheduler(args.scheduler, optimizer, T_max=64, eta_min=1e-5)
    else:
        scheduler = build_scheduler(args.scheduler, optimizer)

    return net, out_idx, criterion, optimizer, scheduler




def prepare_loaders(args, bl_fnc):
    train_loader, query_loader, dbase_loader = (
        bl_fnc(
            args.data_dir,
            args.dataset,
            "train",
            None,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            drop_last=True,
        ),
        bl_fnc(
            args.data_dir,
            args.dataset,
            "query",
            None,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
        ),
        bl_fnc(
            args.data_dir,
            args.dataset,
            "dbase",
            None,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
        ),
    )
    return train_loader, query_loader, dbase_loader




if __name__ == "__main__":
    main()
