from tkinter import N
import torch
from torch import nn
from timm.models.vision_transformer import trunc_normal_
from torch.nn.functional import one_hot

from .patch import PatchEmbedding
from .mask import MaskGenerator
from .positional_encoding import PositionalEncoding
from .transformer_layers import TransformerLayers

def batch_cosine_similarity(x, y):
    # 计算分母
    l2_x = torch.norm(x, dim=2, p=2) + 1e-7  # avoid 0, l2 norm, num_heads x batch_size x hidden_dim==>num_heads x batch_size
    l2_y = torch.norm(y, dim=2, p=2) + 1e-7  # avoid 0, l2 norm, num_heads x batch_size x hidden_dim==>num_heads x batch_size
    l2_m = torch.matmul(l2_x.unsqueeze(dim=2), l2_y.unsqueeze(dim=2).transpose(1, 2))
    # 计算分子
    l2_z = torch.matmul(x, y.transpose(1, 2))
    # cos similarity affinity matrix
    cos_affnity = l2_z / l2_m
    adj = cos_affnity
    return adj

def unshuffle(shuffled_tokens):
    dic = {}
    for k, v, in enumerate(shuffled_tokens):
        dic[v] = k
    unshuffle_index = []
    for i in range(len(shuffled_tokens)):
        unshuffle_index.append(dic[i])
    return unshuffle_index


class TSFormer(nn.Module):
    """An efficient unsupervised pre-training model for Time Series based on transFormer blocks. (TSFormer)"""

    def __init__(self, patch_size, in_channel, embed_dim, num_heads, mlp_ratio, dropout, num_token, mask_ratio, encoder_depth, decoder_depth,
                       mode="pre-train", mask_last_token=False, pretrain_path="", requires_grad=True, decoding_knn=0, strict=True, decoding_knn_node=0):
        super().__init__()
        assert mode in ["pre-train", "forecasting", "3d-finetune"], "Error mode."
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_token = num_token
        self.mask_ratio = mask_ratio
        self.encoder_depth = encoder_depth
        self.mode = mode
        self.mlp_ratio = mlp_ratio

        self.selected_feature = 0

        # norm layers
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)

        # encoder specifics
        # # patchify & embedding
        self.patch_embedding = PatchEmbedding(patch_size, in_channel, embed_dim, norm_layer=None)
        # # positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=dropout)
        # # masking
        self.mask = MaskGenerator(num_token, mask_ratio, mask_last_token=mask_last_token)
        # encoder
        self.encoder = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)

        # decoder specifics
        # transform layer
        self.enc_2_dec_emb = nn.Linear(embed_dim, embed_dim, bias=True)
        # # mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        # neigbor token
        self.decoding_knn = decoding_knn
        self.decoding_knn_node = decoding_knn_node if decoding_knn_node else decoding_knn
        if self.decoding_knn > 0:
            self.neighbor_token = nn.Parameter(torch.zeros(1, 1, self.decoding_knn, embed_dim))
            trunc_normal_(self.neighbor_token)
        self.neighbor_token = nn.Parameter(torch.zeros(1, 1, )) 
        # # decoder
        self.decoder = TransformerLayers(embed_dim, decoder_depth, mlp_ratio, num_heads, dropout)

        # # prediction (reconstruction) layer
        self.output_layer = nn.Linear(embed_dim, patch_size)
        self.initialize_weights()
        if pretrain_path:
            self.load_pre_trained_model(pretrain_path=pretrain_path, requires_grad=requires_grad, strict=strict)


    def load_pre_trained_model(self, pretrain_path="", requires_grad=False, strict=False):
        """Load pre-trained model"""
        print("load pretrained tsformer from: {}".format(pretrain_path))
        # load parameters
        checkpoint_dict = torch.load(pretrain_path)
        self.load_state_dict(checkpoint_dict["model_state_dict"], strict=strict)
        # freeze parameters
        for param in self.parameters():
            param.requires_grad = requires_grad

    def initialize_weights(self):
        # positional encoding
        nn.init.uniform_(self.positional_encoding.position_embedding, -.02, .02)
        # mask token
        trunc_normal_(self.mask_token, std=.02)

    def encoding(self, long_term_history, mask=True):
        """Encoding process of TSFormer: patchify, positional encoding, mask, Transformer layers.

        Args:
            long_term_history (torch.Tensor): Very long-term historical MTS with shape [B, N, 1, P * L],
                                                which is used in the TSFormer.
                                                P is the number of segments (patches).
            mask (bool): True in pre-training stage and False in forecasting stage.

        Returns:
            torch.Tensor: hidden states of unmasked tokens
            list: unmasked token index
            list: masked token index
        """

        batch_size, num_nodes, _, _ = long_term_history.shape
        # patchify and embed input
        patches = self.patch_embedding(long_term_history)     # B, N, d, P
        patches = patches.transpose(-1, -2)         # B, N, P, d
        # positional embedding
        patches = self.positional_encoding(patches)

        # mask
        if mask:
            unmasked_token_index, masked_token_index = self.mask()
            encoder_input = patches[:, :, unmasked_token_index, :]
        else:
            unmasked_token_index, masked_token_index = None, None
            encoder_input = patches

        # encoding
        hidden_states_unmasked = self.encoder(encoder_input)
        hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, num_nodes, -1, self.embed_dim)

        return hidden_states_unmasked, unmasked_token_index, masked_token_index

    def decoding(self, hidden_states_unmasked, masked_token_index=None):
        """Decoding process of TSFormer: encoder 2 decoder layer, add mask tokens, Transformer layers, predict.

        Args:
            hidden_states_unmasked (torch.Tensor): hidden states of masked tokens [B, N, P*(1-r), d].
            masked_token_index (list): masked token index

        Returns:
            torch.Tensor: reconstructed data
        """
        batch_size, num_nodes, unmask_len, embedding_dim = hidden_states_unmasked.shape
        if self.decoding_knn > 0:
            hidden_states_full_sim = hidden_states_unmasked.detach().contiguous().view(batch_size, num_nodes, -1) #B, N, P * (1-r) * d
            batch_sim = batch_cosine_similarity(hidden_states_full_sim, hidden_states_full_sim)
            mask = torch.eye(num_nodes, num_nodes).unsqueeze(0).bool().to(batch_sim.device)
            batch_sim.masked_fill_(mask, 0)
            topk, topk_indices = torch.topk(batch_sim, self.decoding_knn_node, dim=-1)
            topk_onehot = one_hot(topk_indices, num_nodes) # B, N, K, N
            knn_hidden_states_full_sim = torch.bmm(topk_onehot.view(batch_size, -1, num_nodes).float(), hidden_states_full_sim) # B, N, K, P * (1-r) * d selected the topk node
            knn_hidden_states_full_sim = knn_hidden_states_full_sim.view(batch_size, num_nodes, self.decoding_knn_node, -1, embedding_dim) # B, N, K, P * (1-r), d
            knn_hidden_states_full_sim = knn_hidden_states_full_sim.view(batch_size * num_nodes, self.decoding_knn_node * unmask_len, embedding_dim) # B*N, K*l, d
            sub_batch_sim = batch_cosine_similarity(hidden_states_unmasked[:, :, -1, :].detach().contiguous().view(batch_size*num_nodes, 1, embedding_dim), #N, B, d
                                                    knn_hidden_states_full_sim) # B, N, 1, K * P
            sub_topk, sub_topk_indices = torch.topk(sub_batch_sim, self.decoding_knn, dim=-1) # B, N, 1, K
            sub_topk_onehot = one_hot(sub_topk_indices, unmask_len * self.decoding_knn_node) # B, N, 1, K, K*l
            sub_knn_hidden_states = torch.bmm(sub_topk_onehot.view(batch_size * num_nodes, self.decoding_knn, self.decoding_knn_node * unmask_len).float(),
                                              knn_hidden_states_full_sim) # B*N, K, d select the topk patches
            sub_knn_hidden_states = sub_knn_hidden_states.view(batch_size, num_nodes, self.decoding_knn, embedding_dim).to(hidden_states_unmasked.device)




        # encoder 2 decoder layer
        hidden_states_unmasked = self.enc_2_dec_emb(hidden_states_unmasked)

        # add mask tokens
        if masked_token_index is not None:
            hidden_states_masked = self.positional_encoding(
                self.mask_token.expand(batch_size, num_nodes, len(masked_token_index), hidden_states_unmasked.shape[-1]),
                index=masked_token_index
                )
            hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)   # B, N, P, d
        else:
            hidden_states_full = hidden_states_unmasked

        batch_size, num_nodes, full_len, embedding_dim = hidden_states_full.size()
        if self.decoding_knn > 0:
            sub_knn_hidden_states = self.enc_2_dec_emb(sub_knn_hidden_states) #has grad
            sub_knn_hidden_states = self.neighbor_token.expand(batch_size, num_nodes, self.decoding_knn, embedding_dim) + sub_knn_hidden_states
            hidden_states_full = torch.cat([hidden_states_full, sub_knn_hidden_states], dim=-2)
        # decoding
        hidden_states_full = self.decoder(hidden_states_full)
        hidden_states_full = self.decoder_norm(hidden_states_full)

        hidden_states_full = hidden_states_full[:, :, :full_len, :]
        # prediction (reconstruction)
        reconstruction_full = self.output_layer(hidden_states_full.view(batch_size, num_nodes, -1, self.embed_dim))

        return reconstruction_full, hidden_states_full

    def get_reconstructed_masked_tokens(self, reconstruction_full, real_value_full, unmasked_token_index, masked_token_index):
        """Get reconstructed masked tokens and corresponding ground-truth for subsequent loss computing.

        Args:
            reconstruction_full (torch.Tensor): reconstructed full tokens.
            real_value_full (torch.Tensor): ground truth full tokens.
            unmasked_token_index (list): unmasked token index.
            masked_token_index (list): masked token index.

        Returns:
            torch.Tensor: reconstructed masked tokens.
            torch.Tensor: ground truth masked tokens.
        """
        # get reconstructed masked tokens
        batch_size, num_nodes, _, _ = reconstruction_full.shape
        reconstruction_masked_tokens = reconstruction_full[:, :, len(unmasked_token_index):, :]     # B, N, r*P, d
        reconstruction_masked_tokens = reconstruction_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)     # B, r*P*d, N

        label_full = real_value_full.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size)[:, :, :, self.selected_feature, :].transpose(1, 2)  # B, N, P, L
        label_masked_tokens = label_full[:, :, masked_token_index, :].contiguous() # B, N, r*P, d
        label_masked_tokens = label_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)  # B, r*P*d, N

        return reconstruction_masked_tokens, label_masked_tokens

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None, epoch: int = None, **kwargs) -> torch.Tensor:
        """feed forward of the TSFormer.
            TSFormer has two modes: the pre-training mode and the forecasting mode,
                                    which are used in the pre-training stage and the forecasting stage, respectively.

        Args:
            history_data (torch.Tensor): very long-term historical time series with shape B, L * P, N, 1.

        Returns:
            pre-training:
                torch.Tensor: the reconstruction of the masked tokens. Shape [B, L * P * r, N, 1]
                torch.Tensor: the ground truth of the masked tokens. Shape [B, L * P * r, N, 1]
                dict: data for plotting.
            forecasting:
                torch.Tensor: the output of TSFormer of the encoder with shape [B, N, L, 1].
        """
        # reshape
        history_data = history_data.permute(0, 2, 3, 1)     # B, N, 1, L * P
        # feed forward
        if self.mode == "pre-train":
            # encoding
            hidden_states_unmasked, unmasked_token_index, masked_token_index = self.encoding(history_data)
            # decoding
            reconstruction_full, hidden_states_full = self.decoding(hidden_states_unmasked, masked_token_index)
            # for subsequent loss computing
            reconstruction_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(reconstruction_full, history_data, unmasked_token_index, masked_token_index)
            return reconstruction_masked_tokens, label_masked_tokens
        elif self.mode == "3d-finetune":
            hidden_states_unmasked, unmasked_token_index, masked_token_index = self.encoding(history_data)
            # decoding
            reconstruction_full, hidden_states_full_decoding = self.decoding(hidden_states_unmasked, masked_token_index=masked_token_index)
            return hidden_states_unmasked, hidden_states_full_decoding
        else:
            hidden_states_full, _, _ = self.encoding(history_data, mask=False)
            return hidden_states_full, None
