import os
import sys

FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(FILE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger
from yolox.evaluators import MOTEvaluator

import argparse
import random
import warnings
import glob
import motmetrics as mm
from collections import OrderedDict
from pathlib import Path


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--local_rank", default=0, type=int, help="local rank for dist training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    # det args
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=0.01, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--slstm_ckpt", type=str, default=None, help="optional sLSTM token trajectory checkpoint")
    parser.add_argument("--slstm_vocab_size", type=int, default=256, help="sLSTM trajectory token vocabulary size")
    parser.add_argument("--slstm_context_length", type=int, default=256, help="sLSTM token context length")
    parser.add_argument("--slstm_alpha0", type=float, default=0.5, help="maximum sLSTM/Kalman blend weight")
    parser.add_argument("--slstm_beta", type=float, default=0.3, help="sLSTM blend decay for missing tracks")
    parser.add_argument("--xlstm_motion_ckpt", type=str, default=None, help="optional xLSTM motion residual checkpoint")
    parser.add_argument("--xlstm_history_len", type=int, default=16, help="xLSTM motion history length")
    parser.add_argument("--xlstm_input_dim", type=int, default=12, help="xLSTM motion history feature dimension")
    parser.add_argument("--xlstm_min_history", type=int, default=16, help="minimum history length before applying xLSTM")
    parser.add_argument("--xlstm_embedding_dim", type=int, default=128, help="xLSTM motion embedding dimension")
    parser.add_argument("--xlstm_num_blocks", type=int, default=4, help="number of xLSTM blocks")
    parser.add_argument("--xlstm_num_heads", type=int, default=4, help="number of xLSTM heads")
    parser.add_argument("--xlstm_backend", type=str, default="cuda", help="xLSTM sLSTM backend")
    parser.add_argument("--xlstm_device", type=str, default=None, help="device for xLSTM motion model")
    parser.add_argument("--xlstm_covariance_scale", type=float, default=1.0, help="scale for log_var covariance inflation")
    parser.add_argument("--xlstm_max_abs_residual", type=float, default=256.0, help="clip xLSTM residual magnitude")
    parser.add_argument("--ltc_motion_ckpt", type=str, default=None, help="optional LTC/CfC motion residual checkpoint")
    parser.add_argument("--ltc_history_len", type=int, default=16, help="LTC motion history length")
    parser.add_argument("--ltc_input_dim", type=int, default=12, help="LTC motion history feature dimension")
    parser.add_argument("--ltc_min_history", type=int, default=16, help="minimum history length before applying LTC")
    parser.add_argument("--ltc_hidden_size", type=int, default=128, help="LTC hidden size")
    parser.add_argument("--ltc_num_layers", type=int, default=2, help="number of LTC/CfC layers")
    parser.add_argument("--ltc_device", type=str, default=None, help="device for LTC motion model")
    parser.add_argument("--ltc_covariance_scale", type=float, default=1.0, help="scale for LTC log_var covariance inflation")
    parser.add_argument("--ltc_max_abs_residual", type=float, default=256.0, help="clip LTC residual magnitude")
    # reid args
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="use ReID features in ByteTrack association")
    parser.add_argument("--fast-reid", dest="fast_reid", default=False, action="store_true", help="use FastReID backend for ReID features")
    parser.add_argument("--reid-device", type=str, default="cuda", help="ReID device, e.g. cuda or cpu")
    parser.add_argument("--reid-weight", type=float, default=0.35, help="appearance cost weight when fusing IoU and ReID")
    parser.add_argument("--reid-thresh", type=float, default=0.7, help="max cosine distance allowed before ReID cost is capped")
    parser.add_argument("--reid-alpha", type=float, default=0.9, help="EMA momentum for track ReID features")
    parser.add_argument("--reid-model", type=str, default="osnet_x1_0", help="torchreid model name (deep backend)")
    parser.add_argument("--reid-model-path", type=str, default="", help="path to ReID model weights (deep backend)")
    parser.add_argument("--fast-reid-config", type=str, default="", help="FastReID config yaml")
    parser.add_argument("--fast-reid-weights", type=str, default="", help="FastReID model weights; falls back to --reid-model-path")
    parser.add_argument("--fast-reid-batch-size", type=int, default=16, help="FastReID inference batch size")
    # DMA args
    parser.add_argument("--dma-weights", type=str, default=None,
                        help="Path to trained DynamicWeightNet checkpoint (.pth). "
                             "Enables adaptive motion-appearance fusion.")
    parser.add_argument("--dma-device", type=str, default="cuda",
                        help="Device for DMA inference (cpu or cuda)")
    return parser


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:            
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names


@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    cudnn.benchmark = True

    rank = args.local_rank
    # rank = get_local_rank()

    file_name = os.path.join(exp.output_dir, args.experiment_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    results_folder = os.path.join(file_name, "track_results")
    os.makedirs(results_folder, exist_ok=True)
    video_folder = os.path.join(file_name, "videos")

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    #logger.info("Model Structure:\n{}".format(str(model)))

    val_loader = exp.get_eval_loader(args.batch_size, is_distributed, args.test)
    evaluator = MOTEvaluator(
        args=args,
        dataloader=val_loader,
        img_size=exp.test_size,
        confthre=exp.test_conf,
        nmsthre=exp.nmsthre,
        num_classes=exp.num_classes,
        )

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    if not args.speed and not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc)
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert (
            not args.fuse and not is_distributed and args.batch_size == 1
        ), "TensorRT model is not support model fusing and distributed inferencing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    # start evaluate
    *_, summary = evaluator.evaluate(
        model, is_distributed, args.fp16, trt_file, decoder, exp.test_size, results_folder
    )
    logger.info("\n" + summary)

    # evaluate MOTA
    mm.lap.default_solver = 'lap'

    if exp.val_ann == 'val_half.json':
        gt_type = '_val_half'
    else:
        gt_type = ''
    eval_dataset = getattr(val_loader, "dataset", None)
    gt_root = None
    dataset_root = getattr(eval_dataset, "data_dir", None)
    dataset_split = getattr(eval_dataset, "name", None)
    if dataset_root and dataset_split:
        gt_root = os.path.join(dataset_root, dataset_split)

    if args.mot20:
        gt_root = gt_root or os.path.join('datasets', 'MOT20', 'train')
    else:
        gt_root = gt_root or os.path.join('datasets', 'mot', 'train')

    gtfiles = glob.glob(os.path.join(gt_root, '*/gt/gt{}.txt'.format(gt_type)))
    tsfiles = [f for f in glob.glob(os.path.join(results_folder, '*.txt')) if not os.path.basename(f).startswith('eval')]

    logger.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
    logger.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
    logger.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
    logger.info('Loading files.')

    mot_csv_sep = r'\s*,\s*'
    gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1, sep=mot_csv_sep)) for f in gtfiles])
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1, sep=mot_csv_sep)) for f in tsfiles])
    
    mh = mm.metrics.create()    
    accs, names = compare_dataframes(gt, ts)
    
    logger.info('Running metrics')
    metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
               'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
               'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    # summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
    # print(mm.io.render_summary(
    #   summary, formatters=mh.formatters, 
    #   namemap=mm.io.motchallenge_metric_names))
    div_dict = {
        'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
        'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']}
    for divisor in div_dict:
        for divided in div_dict[divisor]:
            summary[divided] = (summary[divided] / summary[divisor])
    fmt = mh.formatters
    change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mostly_tracked',
                       'partially_tracked', 'mostly_lost']
    for k in change_fmt_list:
        fmt[k] = fmt['mota']
    print(mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names))

    metrics = mm.metrics.motchallenge_metrics + ['num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    logger.info('Completed')


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    if args.fast_reid:
        if not args.fast_reid_config:
            parser.error("--fast-reid-config is required with --fast-reid")
        if not args.fast_reid_weights and not args.reid_model_path:
            parser.error("--fast-reid-weights or --reid-model-path is required with --fast-reid")
    args.reid_backend = "fast" if args.fast_reid else "deep"
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        if args.slstm_ckpt:
            args.experiment_name = "eval_slstm"
        elif args.xlstm_motion_ckpt:
            args.experiment_name = "eval_xlstm"
        elif args.ltc_motion_ckpt:
            args.experiment_name = "eval_ltc"
        elif args.with_reid:
            args.experiment_name = "eval_reid"
        else:
            args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args, num_gpu),
    )
