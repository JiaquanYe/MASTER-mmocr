import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import uniform_init, xavier_init

import mmocr.utils as utils
from mmocr.models.builder import ENCODERS
from .base_encoder import BaseEncoder


@ENCODERS.register_module()
class SAREncoder(BaseEncoder):
    """Implementation of encoder module in `SAR.

    <https://arxiv.org/abs/1811.00751>`_

    Args:
        enc_bi_rnn (bool): If True, use bidirectional RNN in encoder.
        enc_do_rnn (float): Dropout probability of RNN layer in encoder.
        enc_gru (bool): If True, use GRU, else LSTM in encoder.
        d_model (int): Dim of channels from backbone.
        d_enc (int): Dim of encoder RNN layer.
        mask (bool): If True, mask padding in RNN sequence.
    """

    def __init__(self,
                 enc_bi_rnn=False,
                 enc_do_rnn=0.0,
                 enc_gru=False,
                 d_model=512,
                 d_enc=512,
                 mask=True,
                 **kwargs):
        super().__init__()
        assert isinstance(enc_bi_rnn, bool)
        assert isinstance(enc_do_rnn, (int, float))
        assert 0 <= enc_do_rnn < 1.0
        assert isinstance(enc_gru, bool)
        assert isinstance(d_model, int)
        assert isinstance(d_enc, int)
        assert isinstance(mask, bool)

        self.enc_bi_rnn = enc_bi_rnn
        self.enc_do_rnn = enc_do_rnn
        self.mask = mask

        # LSTM Encoder
        kwargs = dict(
            input_size=d_model,
            hidden_size=d_enc,
            num_layers=2,
            batch_first=True,
            dropout=enc_do_rnn,
            bidirectional=enc_bi_rnn)
        if enc_gru:
            self.rnn_encoder = nn.GRU(**kwargs)
        else:
            self.rnn_encoder = nn.LSTM(**kwargs)

        # global feature transformation
        encoder_rnn_out_size = d_enc * (int(enc_bi_rnn) + 1)
        self.linear = nn.Linear(encoder_rnn_out_size, encoder_rnn_out_size)

    def init_weights(self):
        # initialize weight and bias
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                uniform_init(m)

    def forward(self, feat, img_metas=None):
        if img_metas is not None:
            assert utils.is_type_list(img_metas, dict)
            assert len(img_metas) == feat.size(0)

        valid_ratios = None
        if img_metas is not None:
            valid_ratios = [
                img_meta.get('valid_ratio', 1.0) for img_meta in img_metas
            ] if self.mask else None

        h_feat = feat.size(2)
        feat_v = F.max_pool2d(
            feat, kernel_size=(h_feat, 1), stride=1, padding=0)
        feat_v = feat_v.squeeze(2)  # bsz * C * W
        feat_v = feat_v.permute(0, 2, 1).contiguous()  # bsz * W * C

        holistic_feat = self.rnn_encoder(feat_v)[0]  # bsz * T * C

        if valid_ratios is not None:
            valid_hf = []
            T = holistic_feat.size(1)
            for i, valid_ratio in enumerate(valid_ratios):
                valid_step = min(T, math.ceil(T * valid_ratio)) - 1
                valid_hf.append(holistic_feat[i, valid_step, :])
            valid_hf = torch.stack(valid_hf, dim=0)
        else:
            valid_hf = holistic_feat[:, -1, :]  # bsz * C

        holistic_feat = self.linear(valid_hf)  # bsz * C

        return holistic_feat
