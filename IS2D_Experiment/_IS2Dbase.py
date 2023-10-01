import os
import sys
from datetime import datetime
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from torch.utils.data import DataLoader

from IS2D_models import IS2D_model, model_to_device
from dataset.BioMedicalDataset.PolypSeg import *
from utils.get_functions import *
from utils.misc import UnNormalize, CSVLogger

class BaseSegmentationExperiment(object):
    def __init__(self, args):
        super(BaseSegmentationExperiment, self).__init__()

        self.args = args

        self.args.device = get_deivce()
        self.fix_seed(self.args.device)
        self.history_generator()
        self.scaler = torch.cuda.amp.GradScaler()
        self.start, self.end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.inference_time_list = []
        self.metric_list = ['Loss', 'Accuracy', 'F1 Score', 'Precision', 'Recall', 'IoU']
        self.unnormalize = UnNormalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                       std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        print("STEP1-1. Load {} Train Dataset Loader...".format(args.train_data_type))
        train_dataset = PolypImageSegDataset(args.train_dataset_dir, mode='train',
                                             transform=self.transform_generator('train', args.train_data_type)[0],
                                             target_transform=self.transform_generator('train', args.train_data_type)[1])

        print("STEP1-2. Load {} Test Dataset Loader...".format(args.test_data_type))
        val_dataset = PolypImageSegDataset(args.test_dataset_dir, mode='val',
                                           transform=self.transform_generator('test', args.test_data_type)[0],
                                           target_transform=self.transform_generator('test', args.test_data_type)[1])
        test_dataset = PolypImageSegDataset(args.test_dataset_dir, mode='test',
                                            transform=self.transform_generator('test', args.test_data_type)[0],
                                            target_transform=self.transform_generator('test', args.test_data_type)[1])

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=int(args.batch_size / args.ngpus_per_node) if args.distributed else args.batch_size,
                                       shuffle=(self.train_sampler is None),
                                       num_workers=int((args.num_workers+args.ngpus_per_node-1)/args.ngpus_per_node) if args.distributed else args.num_workers,
                                       pin_memory=True, sampler=self.train_sampler)
        self.val_loader = DataLoader(val_dataset,
                                      batch_size=int(args.batch_size / args.ngpus_per_node) if args.distributed else args.batch_size,
                                      shuffle=False,
                                      num_workers=int((args.num_workers+args.ngpus_per_node-1)/args.ngpus_per_node) if args.distributed else args.num_workers,
                                      pin_memory=True)
        self.test_loader = DataLoader(test_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=int((args.num_workers+args.ngpus_per_node-1)/args.ngpus_per_node) if args.distributed else args.num_workers,
                                      pin_memory=True)

        print("STEP2. Load 2D Image Segmentation Model {}...".format(args.model_name))
        self.model = IS2D_model(args.model_name, args.image_size, args.num_channels, args.num_classes, args.device, args.backbone_name, args.group, args.inter_channels)
        if self.args.train: self.model = model_to_device(self.args, self.model)

        print("STEP3. Load Optimizer {}...".format(args.optimizer_name))
        self.optimizer = get_optimizer(args.optimizer_name, self.model, args.lr, args.momentum, args.weight_decay)

        print("STEP4. Load LRS {}...".format(args.LRS_name))
        self.scheduler = get_scheduler(args.LRS_name, self.optimizer, args.final_epoch, len(self.train_loader), args.lr)

        print("STEP5. Load Criterion {}...".format(args.criterion))
        self.criterion = get_criterion(args.criterion, ignore_index=args.ignore_index)

        if args.distributed: self.criterion.cuda(args.gpu)

        if self.args.train:
            print("STEP6. Make Train Log File...")
            now = datetime.now()
            model_dirs = get_save_path(args)
            if not os.path.exists(os.path.join(model_dirs, 'logs')): os.makedirs(os.path.join(model_dirs, 'logs'))
            filename = os.path.join(model_dirs, 'logs', '{}-{}-{} {}:{}:{} log.csv'.format(now.year, now.month, now.day, now.hour, now.minute, now.second))
            self.csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_loss', 'test_loss'], filename=filename)

    def print_params(self):
        print("\ntrain data type : {}".format(self.args.train_data_type))
        print("test data type : {}".format(self.args.test_data_type))
        print("model : {}".format(self.args.model_name))
        print("optimizer : {}".format(self.optimizer))
        print("learning rate : {}".format(self.args.lr))
        print("learning rate scheduler : {}".format(self.args.LRS_name))
        print("start epoch : {}".format(self.args.start_epoch))
        print("final epoch : {}".format(self.args.final_epoch))
        print("criterion : {}".format(self.criterion))
        print("batch size : {}".format(self.args.batch_size))
        print("image size : ({}, {}, {})".format(self.args.image_size, self.args.image_size, self.args.num_channels))
        print("backbone : {}".format(self.args.backbone_name))
        print("our_model_save_path : {}".format(self.args.our_model_save_path))

    def fix_seed(self, device):
        random.seed(4321)
        np.random.seed(4321)
        torch.manual_seed(4321)
        torch.cuda.manual_seed(4321)
        torch.cuda.manual_seed_all(4321)
        if self.args.reproducibility :
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print("your seed is fixed to '4321' with reproducibility {}".format(self.args.reproducibility))

    def current_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def forward(self, image, target, mode):
        if self.args.distributed: image, target = image.cuda(), target.cuda()
        else: image, target = image.to(self.args.device), target.to(self.args.device)

        output = self.model(image, mode)
        loss = self.model.module._calculate_criterion(self.criterion, output, target, mode) if torch.cuda.device_count() > 1 else self.model._calculate_criterion(self.criterion, output, target, mode)

        return loss, output, target

    def backward(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.args.LRS_name == 'CALRS': self.scheduler.step()

    def history_generator(self):
        self.history = dict()
        self.history['train_loss'] = list()
        self.history['val_loss'] = list()