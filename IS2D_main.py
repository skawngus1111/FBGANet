#-*- coding:utf-8 -*-
# Distributed Data Parallel 코드
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import argparse
import builtins
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

from utils.get_functions import get_save_path
from utils.save_functions import save_result, save_metrics
from IS2D_Experiment.biomedical_2dimage_segmentation_experiment import BMISegmentationExperiment

def IS2D_main(args) :
    print("Hello! We start experiment for 2D Image Segmentation!")
    print("Distributed Data Parallel {}".format(args.multiprocessing_distributed))

    try:
        args.train_dataset_dir = os.path.join(args.data_path, args.train_data_type)
        args.test_dataset_dir = os.path.join(args.data_path, args.test_data_type)
    except TypeError:
        print("join() argument must be str, bytes, or os.PathLike object, not 'NoneType'")
        print("Please explicitely write the dataset type")
        sys.exit()

    args.image_size = 256
    args.num_channels = 3
    args.num_classes = 1
    args.inter_channels = 512 if args.backbone_name == 'resnet18' else 2048


    args.distributed = False
    if args.multiprocessing_distributed and args.train:
        args.distributed = args.world_size > 1 or args.multiprocessing_distributed
        ngpus_per_node = torch.cuda.device_count()
        args.world_size = ngpus_per_node * args.world_size

        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else :
        experiment = BMISegmentationExperiment(args)

        if args.train:
            model, optimizer, scheduler, history, test_result = experiment.fit()
            save_result(args, model, optimizer, scheduler, history, test_result, args.final_epoch)
        else:
            test_results = experiment.fit()
            model_dirs = get_save_path(args)

            print("Save {} Model Test Results...".format(args.model_name))
            save_metrics(args, test_results, model_dirs, args.final_epoch)

def main_worker(gpu,ngpus_per_node, args):
    # 내용1 :gpu 설정
    args.gpu = gpu
    args.ngpus_per_node = ngpus_per_node

    if args.multiprocessing_distributed and args.gpu !=0:
        def print_pass(*args):
            pass
        builtins.print=print_pass

    if args.gpu is not None: print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url=='env://' and args.rank==-1:
            args.rank=int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank=args.rank*ngpus_per_node + gpu #gpu None아님?
        torch.distributed.init_process_group(backend=args.dist_backend,init_method=args.dist_url, world_size=args.world_size,rank=args.rank)

    experiment = BMISegmentationExperiment(args)

    if args.train:
        model, optimizer, scheduler, history, test_result = experiment.fit()
        save_result(args, model, optimizer, scheduler, history, test_result, args.final_epoch)
    else:
        test_results = experiment.fit()
        model_dirs = get_save_path(args)

        print("Save {} Model Test Results...".format(args.model_name))
        save_metrics(args, test_results, model_dirs, args.final_epoch)

if __name__=='__main__' :
    parser = argparse.ArgumentParser(description='Following are the arguments that can be passed form the terminal itself!')
    parser.add_argument('--data_path', type=str, default='dataset/BioMedicalDataset')
    parser.add_argument('--train_data_type', type=str, required=False, choices=['CVC-ClinicDB'])
    parser.add_argument('--test_data_type', type=str, required=False, choices=['CVC-ClinicDB', 'Kvasir-SEG', 'CVC-300', 'CVC-ColonDB', 'ETIS-LaribPolypDB'])
    parser.add_argument('--model_name', type=str, required=False, choices=['BGANet'])
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--save_path', type=str, default='model_weights')
    parser.add_argument('--save_cpt_interval', type=int, default=None)
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--reproducibility', default=False, action='store_true')

    # Multi-Processing parameters
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')

    # Train parameter
    parser.add_argument('--ignore_index', type=int, default=255)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--criterion', type=str, default='BCE')
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--final_epoch', type=int, default=200)
    parser.add_argument('--crf', default=False, action='store_true')

    # Optimizer Configuration
    parser.add_argument('--optimizer_name', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)

    # Learning Rate Scheduler (LRS) Configuration
    parser.add_argument('--LRS_name', type=str, default=None)

    # Print parameter
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--performance_print_per_epoch', default=False, action='store_true')

    # BGANet Parameters
    parser.add_argument('--backbone_name', type=str, default='res2net50_v1b_26w_4s', choices=['resnet18', 'resnet50',
                                                                                  'res2net50_v1b_26w_4s', 'res2net101_v1b_26w_4s'])
    parser.add_argument('--group', type=int, default=1)
    parser.add_argument('--our_model_save_path', type=str, default='DCT_CDM-FSA-BGCA(RFB)_GAP-BGSA(RFB)_GAP')

    args = parser.parse_args()

    IS2D_main(args)

    # Polyp Segmentation Generalizability Test
    for train_data_type in ['CVC-ClinicDB']:
        args.train_data_type = train_data_type
        for test_data_type in ['Kvasir-SEG', 'CVC-300', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
            args.train = False
            args.test_data_type = test_data_type
            IS2D_main(args)