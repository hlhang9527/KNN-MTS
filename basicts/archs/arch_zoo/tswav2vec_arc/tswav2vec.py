from typing import Optional, Tuple, Union
import torch
from torch import nn
from dataclasses import dataclass
from transformers.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForPreTraining
class TSWav2Vec(Wav2Vec2ForPreTraining):
    def __init__(self, mode="pre-train", conv_dim=(32, 48, 64, 64), conv_stride=(2, 2, 2, 8), conv_kernel=(4, 3, 2, 2)):
        self.mode = mode
        if self.mode == "pre-train":
            mask_time_prob = 0.75
        elif self.mode == "forecasting":
            mask_time_prob = 0
        else:
            assert False, "Wrong mode: {}".format(mode)
        config = Wav2Vec2Config(vocab_size=32, hidden_size=48, num_hidden_layers=4, num_attention_heads=6,
                                intermediate_size=192, hidden_act="gelu",
                                layerdrop=0.1, initializer_range=0.02, layer_norm_eps=1e-5,
                                feat_extract_norm="layer", feat_extract_activation="gelu",
                                conv_dim=conv_dim, conv_stride=conv_stride, conv_kernel=conv_kernel,
                                num_conv_pos_embeddings=12, num_conv_pos_embedding_groups=4,
                                conv_bias=True, do_stable_layer_norm=True,
                                apply_spec_augment=True, mask_time_prob=mask_time_prob, mask_time_length=5, mask_time_min_masks=2,
                                num_codevectors_per_group=100, num_codevector_groups=2, contrastive_logits_temperature=0.1,
                                num_negatives=30, codevector_dim=96, proj_codevector_dim=96, diversity_loss_weight=0.1)
        self.config = config

        super().__init__(config)

