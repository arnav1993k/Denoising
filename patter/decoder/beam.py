import torch
from .base import Decoder


class BeamCTCDecoder(Decoder):
    def __init__(self, labels, lm_path=None, alpha=0, beta=0, cutoff_top_n=40, cutoff_prob=1.0, beam_width=100,
                 num_processes=4, blank_index=0):
        super(BeamCTCDecoder, self).__init__(labels)
        try:
            from ctcdecode import CTCBeamDecoder
        except ImportError:
            raise ImportError("BeamCTCDecoder requires ctcdecode package.")
        self._decoder = CTCBeamDecoder(labels, lm_path, alpha, beta, cutoff_top_n, cutoff_prob, beam_width,
                                       num_processes, blank_index)

    @classmethod
    def from_config(cls, cfg, labels, blank_index=0):
        beam_cfg = {
            "cutoff_top_n": cfg['beam_config']['cutoff_top_n'],
            "cutoff_prob": cfg['beam_config']['cutoff_prob'],
            "beam_width": cfg['beam_config']['beam_width'],
            "num_processes": cfg['num_workers']
        }
        if 'lm' in cfg['beam_config']:
            beam_cfg["lm_path"] = cfg['beam_config']['lm']['lm_path']
            beam_cfg["alpha"] = cfg['beam_config']['lm']['alpha']
            beam_cfg["beta"] = beta=cfg['beam_config']['lm']['beta']
        return cls(labels, blank_index=blank_index, **beam_cfg)

    def convert_to_strings(self, out, seq_len):
        results = []
        for b, batch in enumerate(out):
            utterances = []
            for p, utt in enumerate(batch):
                size = seq_len[b][p]
                if size > 0:
                    transcript = ''.join(map(lambda x: self.int_to_char[x.item()], utt[0:size]))
                else:
                    transcript = ''
                utterances.append(transcript)
            results.append(utterances)
        return results

    @staticmethod
    def convert_tensor(offsets, sizes):
        results = []
        for b, batch in enumerate(offsets):
            utterances = []
            for p, utt in enumerate(batch):
                size = sizes[b][p]
                if sizes[b][p] > 0:
                    utterances.append(utt[0:size])
                else:
                    utterances.append(torch.IntTensor())
            results.append(utterances)
        return results

    def decode(self, probs, sizes=None, num_results=1):
        """
        Decodes probability output using ctcdecode package.
        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes: Size of each sequence in the mini-batch
        Returns:
            string: sequences of the model's best guess for the transcription
        """
        sizes = sizes.squeeze(0) if sizes is not None else None
        out, scores, offsets, seq_lens = self._decoder.decode(probs.cpu(), sizes)

        strings = self.convert_to_strings(out, seq_lens)
        offsets = self.convert_tensor(offsets, seq_lens)
        return strings, offsets, scores
