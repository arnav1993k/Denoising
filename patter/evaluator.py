import math
import time
import torch
from tqdm import tqdm as tqdm_wrap
from marshmallow.exceptions import ValidationError
from torch.utils.data import DataLoader

from patter.config import EvaluatorConfiguration
from patter.data import audio_seq_collate_fn
from patter.decoder import DecoderFactory, GreedyCTCDecoder
from patter.util import AverageMeter, TranscriptionError, split_targets


class Evaluator(object):
    def __init__(self, cfg, tqdm=False, verbose=False):
        self.cfg = cfg
        self.cuda = cfg['cuda']
        self.tqdm = tqdm
        self.verbose = verbose

    def eval(self, model, corpus):
        test_loader = DataLoader(corpus, num_workers=self.cfg['num_workers'], collate_fn=audio_seq_collate_fn,
                                 pin_memory=self.cuda, batch_size=self.cfg['batch_size'])

        if self.cuda:
            model = torch.nn.DataParallel(model.cuda(), dim=1)

        decoder = DecoderFactory.create(self.cfg['decoder'], model.module.labels)
        return validate(test_loader, model, decoder=decoder, tqdm=self.tqdm, verbose=self.verbose)

    @classmethod
    def load(cls, evaluator_config, tqdm=False, verbose=False):
        print(evaluator_config)
        try:
            cfg = EvaluatorConfiguration().load(evaluator_config)
            if len(cfg.errors) > 0:
                raise ValidationError(cfg.errors)
        except ValidationError as err:
            raise err
        return cls(cfg.data, tqdm=tqdm, verbose=verbose)


def validate(val_loader, model, decoder=None, tqdm=True, training=False, log_n_examples=0, verbose=False):
    labels = model.module.labels if type(model) == torch.nn.DataParallel else model.labels
    blank_index = model.module.blank_index if type(model) == torch.nn.DataParallel else model.blank_index
    target_decoder = GreedyCTCDecoder(labels, blank_index=blank_index)
    if decoder is None:
        decoder = target_decoder

    batch_time = AverageMeter()
    losses = AverageMeter()

    model.eval()

    loader = tqdm_wrap(val_loader, desc="Validate", leave=not training) if tqdm else val_loader

    end = time.time()
    err = TranscriptionError()
    examples = []
    for i, data in enumerate(loader):
        err_inst, example = validate_batch(i, data, model, decoder, target_decoder, verbose=verbose, losses=losses if training else None)
        err += err_inst
        if len(examples) < log_n_examples:
            examples.append(example)

        # measure time taken
        batch_time.update(time.time() - end)
        end = time.time()

    if training:
        return err, losses.avg, examples
    return err


def validate_batch(i, data, model, decoder, target_decoder, verbose=False, losses=None):
    loss_fn = model.module.loss if type(model) == torch.nn.DataParallel else model.loss
    is_cuda = model.module.is_cuda if type(model) == torch.nn.DataParallel else model.is_cuda
    # create variables
    feat, target, feat_len, target_len = data
    with torch.no_grad():
        if is_cuda:
            feat = feat.cuda()
            target = target.cpu()
            feat_len = feat_len.cpu()
            target_len = target_len.cpu()

        # compute output
        output, output_len = model(feat, feat_len)
        output_len = output_len.cpu().squeeze(0)

        if losses is not None:
            mb_loss = loss_fn(output, target, output_len, target_len)
            avg_loss = mb_loss.data.sum() / feat.size(0)  # average the loss by minibatch
            inf = math.inf
            if avg_loss == inf or avg_loss == -inf:
                print("WARNING: received an inf loss, setting loss value to 0")
                avg_loss = 0
            losses.update(avg_loss, feat.size(0))

    # do the decode
    decoded_output, _, _ = decoder.decode(output.transpose(0, 1).data, output_len.data)
    target_strings = target_decoder.convert_to_strings(split_targets(target.data, target_len.data))

    example = (decoded_output[0][0], target_strings[0][0])

    err = TranscriptionError()
    for x in range(len(target_strings)):
        transcript, reference = decoded_output[x][0], target_strings[x][0]
        err_inst = TranscriptionError.calculate(transcript, reference)
        err += err_inst
        if verbose:
            print("Ref:", reference.lower())
            print("Hyp:", transcript.lower())
            print("WER:", err_inst.wer, "CER:", err_inst.cer, "\n")

    del output
    del output_len
    return err, example
