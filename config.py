import argparse
import os


def get_config():
    parser = argparse.ArgumentParser(description=os.path.basename(os.path.dirname(__file__)))

    # common settings
    parser.add_argument("--backbone", type=str, default="resnet50", help="see network.py")
    parser.add_argument("--data-dir", type=str, default="../_datasets", help="directory to dataset")
    parser.add_argument("--n-workers", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--n-epochs", type=int, default=100, help="number of epochs to train for")
    parser.add_argument("--batch-size", type=int, default=128, help="input batch size")
    parser.add_argument("--optimizer", type=str, default="amsgrad", help="sgd/rmsprop/adam/amsgrad/adamw")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--wd", type=float, default=0.0, help="weight decay")
    parser.add_argument("--scheduler", type=str, default="reduce", help="none/reduce")
    parser.add_argument("--device", type=str, default="cuda:0", help="device (accelerator) to use")
    parser.add_argument("--parallel-val", type=bool, default=True, help="use a separate thread for validation")

    # changed at runtime
    parser.add_argument("--dataset", type=str, default="cifar", help="cifar/nuswide/flickr/coco")
    parser.add_argument("--n-classes", type=int, default=10, help="number of dataset classes")
    parser.add_argument("--topk", type=int, default=None, help="mAP@topk")
    parser.add_argument("--save-dir", type=str, default="./output", help="directory to output results")
    parser.add_argument("--n-bits", type=int, default=32, help="length of hashing binary")
    parser.add_argument("--n-samples", type=int, default=5000, help="number of training samples")

    # special settings

    # Model parameters
    parser.add_argument("--margin", default=0.25, type=float, help="Rank loss margin")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate")

    # Attention parameters
    parser.add_argument("--img_attention", default=True, type=bool, help="Use self attention on images")
    parser.add_argument("--n_embeds", default=2, type=int, help="Number of embeddings for MIL formulation")

    # Loss weights
    parser.add_argument(
        "--div_weight", default=0.1, type=float, help="Weight term for the log-determinant divergence loss"
    )

    # Training setting
    parser.add_argument("--grad_clip", default=2.0, type=float, help="Gradient clipping threshold")
    parser.add_argument("--warmup", default=50, type=int, help="train the model except for the pretrained CNN weights")

    args = parser.parse_args()

    # mods
    # args.device = "cuda:0"
    # args.parallel_val = False
    args.optimizer = "adamw"
    args.lr = 1e-4
    args.wd = 1e-3
    args.scheduler = "cosine"
    args.warmup = 10
    # args.n_embeds = 4

    return args
