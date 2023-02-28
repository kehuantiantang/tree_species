# coding=utf-8
import importlib
import os
import traceback
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from torch.backends import cudnn
import argparse
import os.path as osp
import random
import shutil
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from tree_species.dataset import TreeSpeciesDataset
from utils.logger import Logger
from utils import AverageMeter, defaultdict, accuracy
from utils.misc import load_pretrained_weights
from utils.util import get_current_lr, beauty_argparse, hyperparams2yaml, get_format_time
from easydict import EasyDict as edict
from utils.gpu_utils import auto_select_gpu
os.environ['CUDA_VISIBLE_DEVICES'] = auto_select_gpu(11000)


class CustomLoss(nn.Module):
    def __init__(self, mode, config):
        super(CustomLoss, self).__init__()
        self.mode = mode
        hyparam = config.loss
        if hyparam.sup_loss == 'ce':
            # self.ce = nn.CrossEntropyLoss()
            self.ce = lambda x, y: nn.CrossEntropyLoss()(x, y)
        elif hyparam.sup_loss == 'bce':
            self.ce = lambda x, y: nn.BCEWithLogitsLoss()(x, F.one_hot(y, num_classes =
            config.loss.num_classes).float())

        self.valid_loss = ['ce', 'total']

        self.weights = hyparam.loss_weights


    def forward(self, input, label = None):
        ce = self.ce(input, label)
        total = self.weights['cls'] * ce
        return total, {'ce':ce, 'total':total}


class Trainer(object):
    def __init__(self, args):

        # 让网络实验可以重复
        random.seed(args.others.seed)
        torch.manual_seed(args.others.seed)
        np.random.seed(args.others.seed)
        torch.cuda.manual_seed_all(args.others.seed)


        self.checkpoint = os.path.join(args.path.checkpoint, args.model.model_name + args.others.comment)

        if args.mode == 'test':
            self.checkpoint = osp.join(osp.split(args.path.resume)[0], 'test')
        elif args.mode == 'pred':
            self.checkpoint = osp.join(osp.split(args.path.resume)[0], 'pred')
        elif args.mode == 'analyse':
            self.checkpoint = osp.join(osp.split(args.path.resume)[0], 'analyse')

        if not os.path.isdir(self.checkpoint):
            os.makedirs(self.checkpoint, exist_ok=True)

        self.backup_path = None
        if args.mode != 'test':
            self.backup_path = hyperparams2yaml(self.checkpoint, args)

        self.args = args

    def get_backup_path(self):
        return self.backup_path


    def train(self, trainloader, model, optimizers, **kwargs):
        # switch to train mode
        model.train()

        losses = {i:AverageMeter() for i in self.criterion.valid_loss}
        accuracy_metric = AverageMeter()
        hyper_loss = self.args.loss

        bar = tqdm(trainloader, total=len(trainloader))
        for batch_idx, (imgs, class_indexes, class_names, stages) in enumerate(bar):
            class_indexes = class_indexes.long().cuda()

            # compute output
            outputs = model(imgs)
            loss, loss_dict = self.criterion(outputs, class_indexes)

            # measure accuracy and record loss
            for k, v in loss_dict.items():
                losses[k].update(v.detach().cpu().numpy(), imgs[0].size(0))

            accuracy_metric.update(accuracy(outputs, class_indexes)[0], imgs[0].size(0))


            # compute gradient and do SGD step
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()


            for optimizer in optimizers:
                optimizer.step()


            # plot progress
            description = 'Epoch:[%03d|%03d]LR:%.3e,Loss:%.4f|top1: %.4f' % (
                kwargs.get('info')[0], kwargs.get('info')[1], kwargs.get('info')[2], losses['total'].avg,
                accuracy_metric.avg)
            bar.set_description(description)

            # if batch_idx == 10:
            #     break
        results = {'train/%s_loss'%k: -v.avg for k, v in losses.items()}

        return {'train/accuracy': accuracy_metric.avg, **results}


    @torch.no_grad()
    def test(self, testloader, model, mode, **kwargs):
        losses = {i:AverageMeter() for i in self.criterion.valid_loss}
        gt_labels, pred_labels = [], []

        # switch to evaluate mode
        model.eval()


        with torch.no_grad():
            bar = tqdm(testloader, total=len(testloader))
            for batch_idx, (imgs, class_indexes, class_names, stages) in enumerate(bar):
                class_indexes = class_indexes.long().cuda()

                # compute output
                outputs = model(imgs)

                # measure accuracy and record loss
                loss, loss_dict = self.criterion(outputs, class_indexes)

                # measure accuracy and record loss
                for k, v in loss_dict.items():
                    losses[k].update(v.detach().cpu().numpy(), imgs[0].size(0))


                gt_labels.extend(class_indexes.detach().cpu().numpy().flatten())
                pred_labels.extend(torch.argmax(outputs, axis = -1).detach().cpu().numpy().flatten())


                # plot progress
                description= f"{mode} loss:{round(losses['total'].avg, 4)}"
                bar.set_description(description)
                # break

        # patch based result
        info = {'%s/%s_loss'%(mode, k): -v.avg for k, v in losses.items()}
        info ={'%s/accuracy'%(mode): accuracy_score(gt_labels, pred_labels), **info}

        Logger.info(classification_report(gt_labels,
                                          pred_labels))

        Logger.debug(confusion_matrix(gt_labels,
                                          pred_labels))
        return info



    def save_checkpoint(self, state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(checkpoint, filename)
        torch.save(state, filepath)

        if is_best:
            shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
            filepath  = os.path.join(checkpoint, 'model_best.pth.tar')


        Logger.debug('Save checkpoint ', filepath, '='*20)


    def prepare_dataset(self):
        Logger.debug('==> Preparing dataset', '='*50)
        hyper_dataset = self.args.dataset



        train_aug = A.Compose([A.RandomBrightnessContrast(p=0.1),
                               A.HorizontalFlip(p=0.5),
                               A.VerticalFlip(p=0.5),
                               A.RandomRotate90(),
                               A.Resize(256, 256),
                               A.RandomCrop(hyper_dataset.img_size,hyper_dataset.img_size),
                               A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                             ToTensorV2()])

        test_aug = A.Compose([A.Resize(hyper_dataset.img_size, hyper_dataset.img_size),
                              A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                            ToTensorV2()])


        train_dataset = TreeSpeciesDataset(hyper_dataset.img_path, hyper_dataset.train_txt, hyper_dataset.class_names, transforms = train_aug)
        Logger.debug('Train dataset:', len(train_dataset))



        val_dataset = TreeSpeciesDataset(hyper_dataset.img_path, hyper_dataset.val_txt, hyper_dataset.class_names, transforms = test_aug)
        Logger.debug('Val dataset:', len(val_dataset))


        test_dataset = TreeSpeciesDataset(hyper_dataset.img_path, hyper_dataset.test_txt, hyper_dataset.class_names, transforms = test_aug)
        Logger.debug('Test dataset:', len(test_dataset))

        train_dataloader = DataLoader(train_dataset, batch_size = self.args.optimizer.train_batch,
                                      num_workers=self.args.others.num_workers, shuffle = True)
        val_dataloader = DataLoader(val_dataset, batch_size = self.args.optimizer.val_batch,
                                    num_workers=self.args.others.num_workers, shuffle=False)

        test_dataloader = DataLoader(test_dataset, batch_size = self.args.optimizer.test_batch,
                                     num_workers=self.args.others.num_workers, shuffle=False)



        return train_dataloader, val_dataloader, test_dataloader


    def build_model_loss(self, **kwargs):
        hyper_models = self.args.model

        print("==> creating model '{}'".format(hyper_models.model_name))
        filename, model_name = hyper_models.model_name.split('.')
        module = getattr(importlib.import_module('models.' + filename), model_name)
        model = module(pretrained=hyper_models.pretrained, progress=True, start_nb_channels=hyper_models.start_nb_channels, num_classes=hyper_models.num_classes,
                       drop_rate=hyper_models.drop_rate)

        # summary network
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0


        self.criterion = CustomLoss(self.args.mode, self.args).to(device)

        return model


    def save_best_result(self, results, save_state_dicts, mode = 'val'):
        flag = False
        for k, v in results.items():
            if self.best_values[k] < v:
                self.best_values[k] = v
                best_metric_name = k.split('/')[-1]
                if best_metric_name in self.args.others.best_save_metrics:
                    filename = '%s_best_%s.pth.tar'%(mode,
                                                     k.replace(
                                                         '/', '-'))
                    self.save_checkpoint(save_state_dicts, True, checkpoint=self.checkpoint, filename=filename)
                    Logger.info('Save best %s, %.3f ========' % (k.replace('/', '-'), v),
                                osp.join(self.checkpoint, filename))
                    flag = True
        return flag


    def main(self):
        self.best_values = defaultdict(lambda : -sys.maxsize - 1)
        trainloader, valloader, testloader= self.prepare_dataset()

        model = self.build_model_loss()

        # multiple gpu
        self.model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True

        Logger.debug('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

        hyper_optimizer = self.args.optimizer


        optimizers = [optim.Adam(model.parameters(), lr=hyper_optimizer.lr[0],weight_decay=hyper_optimizer.weight_decay), ]
        warm_up_with_multistep_lr = lambda \
                epoch: hyper_optimizer.scheduler.warm_up_lr if epoch <= hyper_optimizer.scheduler.warm_up_epochs else 0.1 ** len(
            [m for m in hyper_optimizer.scheduler.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizers[0], lr_lambda=warm_up_with_multistep_lr)

        # scheduler = ReduceLROnPlateau(optimizers[0], 'min', verbose=True,
        #                               patience=hyper_optimizer.scheduler.patience,
        #                               factor=hyper_optimizer.scheduler.factor)

        # warm_up_with_cosine_lr = lambda epoch: hyper_optimizer.scheduler.warm_up_lr_rate if epoch <= hyper_optimizer.scheduler.warm_up_epochs else \
        #     0.5 * ( math.cos((epoch-hyper_optimizer.scheduler.warm_up_epochs) /(hyper_optimizer.scheduler.cycle_epoch -
        #                                                                         hyper_optimizer.scheduler.warm_up_epochs) * math.pi) + 1)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizers[0], lr_lambda=warm_up_with_cosine_lr)


        if self.args.resume:
            self.model = load_pretrained_weights(self.args.resume, self.model, 'state_dict', 'epoch')
            writer = SummaryWriter(os.path.join(self.checkpoint, 'tf'))
            Logger.info('Save record to ', os.path.join(self.checkpoint, 'tf'))
        else:
            # log checkpoint path
            cp = self.checkpoint
            writer = SummaryWriter(os.path.join(cp, 'tf'))

        # different operation
        if self.args.mode == 'eval':
            Logger.debug('Evaluation only')
            self.args.show_info = True
            info = self.test(valloader, self.model, 'eval')
            Logger.info(info)
            return

        elif self.args.mode == 'pred':
            self.predict(testloader, self.model, writer = writer, mode = 'run_pred')
            return

        # start to train
        for epoch in range(self.args.step.start_epoch, self.args.step.epochs+1):
            lr = get_current_lr(optimizers[0])
            results = self.train(trainloader, self.model, optimizers, info = (epoch,
                                                                              self.args.step.epochs, lr))
            scheduler.step()
            Logger.debug('Epoch:%d, %s'%(epoch, results))
            results = {**results, 'lr': lr}
            for key, value in results.items():
                writer.add_scalar('%s' %key,  value, epoch)


            state_dicts = {'epoch': epoch, 'state_dict': self.model.state_dict(), 'optimizer':
                [optimizer.state_dict() for optimizer in optimizers], }

            if epoch % self.args.step.val_epochs == 0 and self.args.step.val_epochs > 0 and epoch > 0:
                Logger.debug('Start val, ', epoch, '*' * 50)

                results = self.test(valloader, self.model, mode = 'val')
                Logger.info(results)


                for key, value in results.items():
                    writer.add_scalar(key, value, epoch)

                is_best = self.save_best_result(results, state_dicts, mode = 'val')

                # if is_best :
                #     results = self.test(testloader, self.model, mode = 'test')
                #     Logger.info(results)

                Logger.debug('*'*100)



            if (epoch) % self.args.step.save_interval == 0 and epoch != 0:
                self.save_checkpoint(state_dicts, False, checkpoint=self.checkpoint,
                                     filename='train_%04d.pth.tar' %
                                              epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch music style training')
    parser.add_argument('-config', help="configuration file *.yml", type=str, default =
    '../config/train_base.yaml')
    parser.add_argument('-logfile_level', default = 'debug')
    parser.add_argument('-stdout_level', default = 'info')

    parser.add_argument('--mode', default="train", type=str)

    parser.add_argument('--resume', default="", type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')



    args = parser.parse_args()

    configs = yaml.unsafe_load(open(args.config, "r"))

    for k, v in vars(args).items():
        configs[k] = v

    configs = edict(configs)
    hyperparams = beauty_argparse(configs, verbose = False)

    trainer = Trainer(configs)
    backup_path = trainer.get_backup_path()

    Logger.init(logfile_level=args.logfile_level, stdout_level=args.stdout_level,
                log_file=osp.join(osp.split(backup_path)[0], '%s.log'%get_format_time()), rewrite=False)

    Logger.info(hyperparams)

    import sys, signal
    def rollback(signal, frame):
        print('Signal:', signal)
        if args.mode == '':
            root, name = osp.split(backup_path)
            for root, _, filenames in os.walk(root):
                for filename in filenames:
                    if name in filename and not filename.endswith('log'):
                        os.remove(osp.join(root, filename))
                        Logger.error('Error rollback, ', osp.join(root, filename))
        sys.exit(0)
    # signal.signal(signal.SIGINT, rollback)

    try:
        trainer.main()
    except Exception:
        Logger.critical(str(traceback.format_exc()))
        rollback(None, None)