import math
import time
import torch
from tqdm import tqdm as tqdm_wrap
from torch.utils.data import DataLoader
from marshmallow.exceptions import ValidationError

from patter.config import TrainerConfiguration
from patter.data import BucketingSampler, audio_seq_collate_fn
from patter.util import AverageMeter, TensorboardLogger
from patter.models import SpeechModel
from patter.evaluator import validate

from apex import amp

optimizers = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam
}


class NoOpScheduler(object):
    def __init__(self):
        pass

    def step(self):
        pass


class Trainer(object):
    def __init__(self, train_config, tqdm=False):
        self.cfg = train_config['trainer']
        self.output = train_config['output']
        self.cuda = train_config['cuda']
        self.train_id = train_config['expt_id']
        self.tqdm = tqdm
        self.max_norm = self.cfg.get('max_norm', None)
        self.logger = TensorboardLogger(train_config['expt_id'], self.output['log_path'], include_grad=True)
        self.fp16 = train_config['fp16']
        self.amp = amp.init(enabled=self.fp16)
        if self.fp16:
            print("Trainer initialized in fp16 mode.")

    def warmup(self, model, corpus, optimizer):
        """
        Find the largest possible minibatch that will be performed during training, and perform a forward/backward
        pass and optimizer step. This allocates much of the memory that will be needed during training and helps
        to reduce fragmentation
        :param model: The model to be trained
        :param corpus: AudioDataset with the training corpus
        :param optimizer: The optimizer to use for the forward step
        :return:
        """
        # warm up with the largest sized minibatch
        feat, target, feat_len, target_len = corpus.get_largest_minibatch(self.cfg['batch_size'])
        if self.cuda:
            feat = feat.cuda(async=True)
        # self.logger.add_graph(model, (feat, feat_len))
        optimizer.zero_grad()
        output, output_len = model(feat, feat_len)
        loss = model.module.loss(output, target, output_len.cpu().squeeze(0), target_len)
        with self.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        del feat
        del loss
        del output
        del output_len

    def train(self, model, corpus, eval_corpus=None):
        """
        Primary training method. Responsible for training the passed in model using data from the supplied corpus
        according to the configuration of the Trainer object.
        :param model: A patter.SpeechModel to train
        :param corpus: AudioDataset with the training corpus
        :param eval_corpus: Optional AudioDataset with a development corpus
        :return:
        """
        # set up data loaders
        train_sampler = BucketingSampler(corpus, batch_size=self.cfg['batch_size'])
        train_loader = DataLoader(corpus, num_workers=self.cfg['num_workers'], collate_fn=audio_seq_collate_fn,
                                  pin_memory=self.cuda, batch_sampler=train_sampler)
        if eval_corpus is not None:
            eval_loader = DataLoader(eval_corpus, num_workers=self.cfg['num_workers'], collate_fn=audio_seq_collate_fn,
                                     pin_memory=self.cuda, batch_size=self.cfg['batch_size'])
        else:
            eval_loader = None

        if self.cuda:
            model = model.cuda()

        print(model)

        # set up optimizer
        opt_cfg = self.cfg['optimizer']
        optim_class = optimizers.get(opt_cfg['optimizer'])
        del opt_cfg['optimizer']
        optimizer = optim_class(model.parameters(), **opt_cfg)
        print("Configured with optimizer:", optimizer)
        # memoize initial optimizer state so we can reset it after warmup
        optim_state_dict = optimizer.state_dict()

        # set up a learning rate scheduler if requested -- currently only StepLR supported
        if "scheduler" in self.cfg:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.cfg['scheduler']['lr_annealing'])
            print("Configured with learning rate scheduler:", scheduler)
        else:
            scheduler = NoOpScheduler()

        # warm up gpu memory cache by doing a fwd/bwd, validation, and resetting model
        print("Starting warmup...")
        self.warmup(model, corpus, optimizer)
        err = validate(eval_loader, model)
        self.logger.log_epoch(0, 500, err.wer, err.cer, 500)
        # initialize model and optimizer properly for real training
        optimizer.load_state_dict(optim_state_dict)
        model.module.init_weights()
        del optim_state_dict
        torch.cuda.synchronize()
        print("Warmup complete.")

        # primary training loop
        best_wer = math.inf

        wers, cers, losses = [], [], []
        for epoch in range(self.cfg['epochs']):
            # shuffle the input data if required
            if epoch > 0:
                train_sampler.shuffle()

            # adjust lr
            scheduler.step()
            # print("> Learning rate annealed to {0:.6f}".format(scheduler.get_lr()[0]))
            
            avg_loss = self.train_epoch(model, train_loader, optimizer, epoch)
            print("Epoch {} Summary:".format(epoch+1))
            print('    Train:\tAverage Loss {loss:.3f}\t'.format(loss=avg_loss))

            err, val_loss, sample_decodes = validate(eval_loader, model, training=True, log_n_examples=10)
            print('    Validation:\tAverage WER {wer:.3f}\tAverage CER {cer:.3f}'
                  .format(wer=err.wer, cer=err.cer))

            # log the result of the epoch
            wers.append(err.wer), cers.append(err.cer), losses.append(avg_loss)
            self.logger.log_epoch(epoch+1, avg_loss, err.wer, err.cer, val_loss, model=model)
            self.logger.log_images(epoch+1, model.module.get_filter_images())
            self.logger.log_sample_decodes(epoch+1, sample_decodes)

            if err.wer < best_wer:
                best_wer = err.wer
                print("Better model found. Saving.")
                torch.save(SpeechModel.serialize(model.module, optimizer=optimizer, epoch=epoch, loss_results=losses,
                                                 cer_results=cers, wer_results=wers), self.output['model_path'])

    def train_epoch(self, model, train_loader, optimizer, epoch):
        """
        Train the passed in model for a single epoch
        :param model: SpeechModel to train
        :param train_loader: DataLoader wrapping an AudioDataset for the training data
        :param optimizer: Initialized optimizer to use
        :param epoch: Current epoch number
        :return:
        """
        model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        loader = train_loader
        if self.tqdm:
            loader = tqdm_wrap(loader, desc="Epoch {}".format(epoch+1), leave=False)

        end = time.time()
        for i, data in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # create variables
            feat, target, feat_len, target_len, _ = data
            if self.cuda:
                feat = feat.cuda(async=True)

            optimizer.zero_grad()

            # compute output
            # feat is (batch, 1,  feat_dim,  seq_len)
            # output is (seq_len, batch, output_dim)
            output, output_len = model(feat, feat_len)
            loss = model.module.loss(output, target, output_len.cpu().squeeze(0), target_len)

            # munge the loss
            scalar_loss = loss.item()/feat.size(0)
            if abs(scalar_loss) == math.inf:
                print("WARNING: received an inf loss, setting loss value to 0")
                scalar_loss = 0
            self.logger.log_step(epoch*len(train_loader) + i, scalar_loss)
            losses.update(scalar_loss, feat.size(0))

            # compute gradient
            with self.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if self.max_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
            optimizer.step()

            del feat
            del scalar_loss
            del loss
            del output
            del output_len

            # measure time taken
            batch_time.update(time.time() - end)
            end = time.time()

            if not self.tqdm:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format((epoch + 1), (i + 1), len(train_loader),
                                                                      batch_time=batch_time, data_time=data_time,
                                                                      loss=losses))
            else:
                loader.set_postfix(loss=losses.val)

            if self.output['checkpoint_interval'] > 0 and (i % self.output['checkpoint_interval']) == 0:
                torch.save(SpeechModel.serialize(model.module, optimizer=optimizer, epoch=epoch, iteration=i), self.output['model_path']+".ckpt")
        return losses.avg

    @classmethod
    def load(cls, trainer_config, tqdm=False):
        """
        Initialize a Trainer object based on a Trainer configuration
        :param trainer_config:
        :param tqdm:
        :return:
        """
        try:
            cfg = TrainerConfiguration().load(trainer_config)
            if len(cfg.errors) > 0:
                raise ValidationError(cfg.errors)
        except ValidationError as err:
            raise err
        return cls(cfg.data, tqdm=tqdm)
