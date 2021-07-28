import torch.nn as nn
from mmcv.cnn import uniform_init, xavier_init
from mmcv.runner import load_checkpoint

from mmdet.utils import get_root_logger
from mmocr.models.builder import ENCODERS
from mmocr.models.ner.utils.bert import BertModel


@ENCODERS.register_module()
class BertEncoder(nn.Module):
    """Bert encoder
    Args:
        num_hidden_layers (int): The number of hidden layers.
        initializer_range (float):
        vocab_size (int): Number of words supported.
        hidden_size (int): Hidden size.
        max_position_embeddings (int): Max positions embedding size.
        type_vocab_size (int): The size of type_vocab.
        layer_norm_eps (float): Epsilon of layer norm.
        hidden_dropout_prob (float): The dropout probability of hidden layer.
        output_attentions (bool):  Whether use the attentions in output.
        output_hidden_states (bool): Whether use the hidden_states in output.
        num_attention_heads (int): The number of attention heads.
        attention_probs_dropout_prob (float): The dropout probability
            of attention.
        intermediate_size (int): The size of intermediate layer.
        hidden_act_cfg (dict):  Hidden layer activation.
    """

    def __init__(self,
                 num_hidden_layers=12,
                 initializer_range=0.02,
                 vocab_size=21128,
                 hidden_size=768,
                 max_position_embeddings=128,
                 type_vocab_size=2,
                 layer_norm_eps=1e-12,
                 hidden_dropout_prob=0.1,
                 output_attentions=False,
                 output_hidden_states=False,
                 num_attention_heads=12,
                 attention_probs_dropout_prob=0.1,
                 intermediate_size=3072,
                 hidden_act_cfg=dict(type='GeluNew'),
                 pretrained=None):
        super().__init__()
        self.bert = BertModel(
            num_hidden_layers=num_hidden_layers,
            initializer_range=initializer_range,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            intermediate_size=intermediate_size,
            hidden_act_cfg=hidden_act_cfg)
        self.init_weights(pretrained=pretrained)

    def forward(self, results):

        device = next(self.bert.parameters()).device
        input_ids = results['input_ids'].to(device)
        attention_masks = results['attention_masks'].to(device)
        token_type_ids = results['token_type_ids'].to(device)

        outputs = self.bert(
            input_ids=input_ids,
            attention_masks=attention_masks,
            token_type_ids=token_type_ids)
        return outputs

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    uniform_init(m)
