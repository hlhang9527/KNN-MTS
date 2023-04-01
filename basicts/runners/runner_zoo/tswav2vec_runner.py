from curses import noecho
import time
from tqdm import tqdm
import torch
from typing import Tuple, Union, Optional, Dict

from easytorch.utils.dist import master_only
from easytorch.utils import TimePredictor, get_local_rank
from ...data.registry import SCALER_REGISTRY
from ...runners import BaseTimeSeriesForecastingRunner
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
from transformers.models.wav2vec2.feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor
from torch.nn.parallel import DistributedDataParallel

class TsWav2VecRunner(BaseTimeSeriesForecastingRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.forward_features = cfg["MODEL"].get("FROWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor(return_attention_mask=True, feature_size=1, padding_value=0)

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select input features and reshape data to fit the target model.

        Args:
            data (torch.Tensor): input history data, shape [B, L, N, C].

        Returns:
            torch.Tensor: reshaped data
        """

        # select feature using self.forward_features
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select target features and reshape data back to the BasicTS framework

        Args:
            data (torch.Tensor): prediction of the model with arbitrary shape.

        Returns:
            torch.Tensor: reshaped data with shape [B, L, N, C]
        """

        # select feature using self.target_features
        data = data[:, :, :, self.target_features]
        return data

    def init_training(self, cfg: dict):
        super().init_training(cfg)
        self.register_epoch_meter("train_loss", "train", "{:.4f}")
        self.register_epoch_meter("train_diversity_loss", "train", "{:.4f}")

    def init_validation(self, cfg: dict):
        super().init_validation(cfg)
        self.register_epoch_meter("val_loss", "val", "{:.4f}")
        self.register_epoch_meter("val_diversity_loss", "val", "{:.4f}")

    def init_test(self, cfg: dict):
        super().init_test(cfg)
        self.register_epoch_meter("test_loss", "test", "{:.4f}")
        self.register_epoch_meter("test_diversity_loss", "test", "{:.4f}")

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """Training details.

        Args:
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader
            epoch (int): current epoch.
            iter_index (int): current iter.

        Returns:
            loss (torch.Tensor)
        """

        iter_num = (epoch-1) * self.iter_per_epoch + iter_index
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)
        self.update_epoch_meter("train_loss", forward_return.loss.item())
        self.update_epoch_meter("train_diversity_loss", forward_return.diversity_loss.item())
        return forward_return[0]

    def forward(self, data: tuple, epoch:int = None, iter_num: int = None, train:bool = True, **kwargs) -> tuple:
        """feed forward process for train, val, and test. Note that the outputs are NOT re-scaled.

        Args:
            data (tuple): data (future data, history data). [B, L, N, C] for each of them
            epoch (int, optional): epoch number. Defaults to None.
            iter_num (int, optional): iteration number. Defaults to None.
            train (bool, optional): if in the training process. Defaults to True.

        Returns:
            tuple: (prediction, real_value)
        """

        # preprocess
        future_data, history_data, idx,  long_history_data = data
 # =======================reformat=====================================================================

        history_data_for_wv = history_data.permute(0, 2, 3, 1)     # B, N, 1, L * P
        history_data_for_wv = history_data_for_wv[:, :, 0, :]
        batch_size, num_node, length = history_data_for_wv.size()

        input_values = history_data_for_wv.contiguous().view(batch_size * num_node, length)
        input_values = self.to_running_device(input_values)

        features = [{"input_values": input_values[i, :]} for i in range(input_values.size()[0])]
        batch = self.wav2vec_feature_extractor.pad(
            features,
            padding="longest",
            pad_to_multiple_of=None,
            return_tensors="pt",
        )
        batch_size = batch["input_values"].shape[0]
        if isinstance(self.model, DistributedDataParallel):
            tmp_model = self.model.module
        else:
            tmp_model = self.model       	
        mask_indices_seq_length = tmp_model._get_feat_extract_output_lengths(batch["input_values"].shape[-1])
        # make sure masked sequence length is a Python scalar
        mask_indices_seq_length = int(mask_indices_seq_length)

        # make sure that no loss is computed on padded inputs
        attention_mask = None
        if batch.get("attention_mask") is not None:
            # compute real output lengths according to convolution formula
            batch["sub_attention_mask"] = tmp_model._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"])
            attention_mask = batch["attention_mask"]
            attention_mask = self.to_running_device(attention_mask)

        features_shape = (batch_size, mask_indices_seq_length)
        # sample negative indices
        sampled_negative_indices = None
        mask_time_indices = None
        if tmp_model.mode == "pre-train":
                    # sample randomly masked indices
            mask_time_indices = _compute_mask_indices(features_shape, tmp_model.config.mask_time_prob, tmp_model.config.mask_time_length, attention_mask=batch.get("sub_attention_mask"))
            sampled_negative_indices = _sample_negative_indices(features_shape, tmp_model.config.num_negatives, mask_time_indices=mask_time_indices)
            sampled_negative_indices = torch.from_numpy(sampled_negative_indices)

            sampled_negative_indices = self.to_running_device(sampled_negative_indices)

            mask_time_indices = torch.from_numpy(mask_time_indices)
            mask_time_indices = self.to_running_device(mask_time_indices)


        # ================================================================================
        history_data = self.select_input_features(history_data)
        history_data    = self.to_running_device(history_data)      # B, L, N, C
        future_data     = self.to_running_device(future_data)       # B, L, N, C
        batch_size, length, num_nodes, _ = future_data.shape

        # feed forward
        tswav2vec_output = self.model(input_values=input_values, attention_mask=attention_mask,
                                      sampled_negative_indices=sampled_negative_indices,
                                      mask_time_indices=mask_time_indices,
                                      output_attentions=False, output_hidden_states=False, return_dict=True)
        mask_time_indices_sum = int(mask_time_indices.sum())
        tswav2vec_output.loss = tswav2vec_output.loss / mask_time_indices_sum

        return tswav2vec_output

    @torch.no_grad()
    @master_only
    def test(self):
        """Evaluate the model.

        Args:
            train_epoch (int, optional): current epoch if in training process.
        """

        for _, data in enumerate(self.test_data_loader):
            forward_return = self.forward(data=data, epoch=None, iter_num=None, train=False)
            self.update_epoch_meter("test_loss", forward_return.loss.item())
            self.update_epoch_meter("test_diversity_loss", forward_return.diversity_loss.item())

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):
        """Validation details.

        Args:
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader
            train_epoch (int): current epoch if in training process. Else None.
            iter_index (int): current iter.
        """

        forward_return = self.forward(data=data, epoch=None, iter_num=None, train=False)
        self.update_epoch_meter("val_loss", forward_return.loss.item())
        self.update_epoch_meter("val_diversity_loss", forward_return.diversity_loss.item())

    @master_only
    def on_validating_end(self, train_epoch: Optional[int]):
        """Callback at the end of validating.

        Args:
            train_epoch (Optional[int]): current epoch if in training process.
        """

        if train_epoch is not None:
            self.save_best_model(train_epoch, "val_loss", greater_best=False)

