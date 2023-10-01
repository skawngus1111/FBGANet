from time import time

import torch
import torchvision.transforms as transforms

import numpy as np
from tqdm import tqdm

from ._IS2Dbase import BaseSegmentationExperiment
from utils.calculate_metrics import metrics
from utils.save_functions import save_result
from utils.load_functions import load_model
from utils.plot_functions import plot_biomedical_image_and_prediction

class BMISegmentationExperiment(BaseSegmentationExperiment):
    def __init__(self, args):
        super(BMISegmentationExperiment, self).__init__(args)

        self.count = 1

    def fit(self):
        self.print_params()
        if self.args.train:
            for epoch in tqdm(range(self.args.start_epoch, self.args.final_epoch + 1)):
                print('\n============ EPOCH {}/{} ============\n'.format(epoch, self.args.final_epoch))
                if self.args.distributed: self.train_sampler.set_epoch(epoch)
                if epoch % 10 == 0: self.print_params()
                if self.args.LRS_name == 'MSLRS' or self.args.LRS_name == 'SLRS': self.scheduler.step()
                epoch_start_time = time()

                print("TRAINING")
                train_results = self.train_epoch(epoch)

                print("EVALUATE")
                val_results = self.val_epoch(epoch)


                total_epoch_time = time() - epoch_start_time
                m, s = divmod(total_epoch_time, 60)
                h, m = divmod(m, 60)

                self.history['train_loss'].append(train_results)
                self.history['val_loss'].append(val_results)

                if self.args.train:
                    row = {'epoch': str(epoch),
                           'train_loss': str(train_results),
                           'test_loss': str(val_results)}
                    self.csv_logger.writerow(row)

                print('\nEpoch {}/{} : train loss {} | val loss {} | current lr {} | took {} h {} m {} s'.format(
                    epoch, self.args.final_epoch, np.round(train_results, 4), np.round(val_results, 4),
                    self.current_lr(self.optimizer), int(h), int(m), int(s)))

                # if self.args.save_cpt_interval is not None and epoch % self.args.save_cpt_interval == 0:
                #     save_result(self.args, self.model, self.optimizer, self.scheduler, self.history, val_results, epoch, best_results=None, metric_list=self.metric_list)

            print("INFERENCE")
            test_results = self.inference(self.args.final_epoch)

            return self.model, self.optimizer, self.scheduler, self.history, test_results
        else :
            print("INFERENCE")
            self.model = load_model(self.args, self.model)
            test_results = self.inference(self.args.final_epoch)

            return test_results

    def train_epoch(self, epoch):
        self.model.train()

        total_loss, total = 0., 0

        for batch_idx, (image, target) in enumerate(self.train_loader):
            loss, output, target = self.forward(image, target, mode='train')
            self.backward(loss)

            total_loss += loss.item() * image.size(0)
            total += image.size(0)

            if (batch_idx + 1) % self.args.step == 0 or (batch_idx + 1) == len(self.train_loader):
                print("Epoch {} | batch_idx : {}/{}({}%) COMPLETE | loss : {}".format(
                    epoch, batch_idx + 1, len(self.train_loader), np.round((batch_idx + 1) / len(self.train_loader) * 100.0, 2),
                    total_loss / total
                ))

        train_loss = total_loss / total

        return train_loss

    def val_epoch(self, epoch):
        self.model.eval()

        total_loss, total = .0, 0

        with torch.no_grad():
            for batch_idx, (image, target) in enumerate(self.val_loader):
                if (batch_idx + 1) % self.args.step == 0:
                    print("EPOCH {} | {}/{}({}%) COMPLETE".format(epoch, batch_idx + 1, len(self.test_loader), np.round((batch_idx + 1) / len(self.test_loader) * 100), 4))

                loss, output, target = self.forward(image, target, mode='val')

                total_loss += loss.item() * image.size(0)
                total += target.size(0)

        val_loss = total_loss / total

        return val_loss

    def inference(self, epoch):
        self.model.eval()

        total_loss, total = .0, 0
        accuracy_list, f1_score_list, precision_list, recall_list, iou_list = [], [], [], [], []

        with torch.no_grad():
            for batch_idx, (image, target) in enumerate(self.test_loader):
                if (batch_idx + 1) % self.args.step == 0:
                    print("EPOCH {} | {}/{}({}%) COMPLETE".format(epoch, batch_idx + 1, len(self.test_loader), np.round((batch_idx + 1) / len(self.test_loader) * 100), 4))

                self.start.record()
                loss, output, target = self.forward(image, target, mode='test')
                self.end.record()
                torch.cuda.synchronize()
                self.inference_time_list.append(self.start.elapsed_time(self.end))

                for idx, (target_, output_) in enumerate(zip(target, output)):
                    predict = torch.sigmoid(output_).squeeze()
                    plot_biomedical_image_and_prediction(self.args, image[idx], target_, predict, self.count)
                    self.count += 1
                    accuracy, f1_score, precision, recall, iou = metrics(target_, predict)
                    accuracy_list.append(accuracy); f1_score_list.append(f1_score); precision_list.append(precision), recall_list.append(recall), iou_list.append(iou)

                total_loss += loss.item() * image.size(0)
                total += target.size(0)

        test_loss = total_loss / total
        accuracy = np.round(np.mean(accuracy_list), 4)
        f1_score = np.round(np.mean(f1_score_list), 4)
        precision = np.round(np.mean(precision_list), 4)
        recall = np.round(np.mean(recall_list), 4)
        iou = np.round(np.mean(iou_list), 4)

        FPS = np.round(len(self.test_loader) / (np.mean(self.inference_time_list) / 1000), 2)

        return test_loss, accuracy, f1_score, precision, recall, iou, FPS

    def transform_generator(self, mode, data_type=None):
        if mode == 'train' :
            transform_list = [
                transforms.RandomCrop((self.args.image_size, self.args.image_size)),
                transforms.RandomRotation((-5, 5), expand=False),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
            ]

            target_transform_list = [
                transforms.RandomCrop((self.args.image_size, self.args.image_size)),
                transforms.RandomRotation((-5, 5), expand=False),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
            ]

        else :
            transform_list = [
                transforms.Resize((self.args.image_size, self.args.image_size)),
                transforms.ToTensor(),
            ]

            target_transform_list = [
                transforms.Resize((self.args.image_size, self.args.image_size)),
                transforms.ToTensor(),

            ]

        if data_type != 'DSB2018':
            transform_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))

        return transforms.Compose(transform_list), transforms.Compose(target_transform_list)