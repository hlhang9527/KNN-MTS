import math
import functools
from select import POLLOUT
from typing import Tuple, Union, Optional
import os
from webbrowser import get
from tqdm import tqdm
import torch
import numpy as np
import time
import json
import pickle
from easytorch.utils.dist import master_only

import matplotlib.pyplot as plt

from .base_runner import BaseRunner
from ..data import SCALER_REGISTRY
from ..utils import load_pkl
from ..metrics import masked_mae, masked_mape, masked_rmse
from ..data import SCALER_REGISTRY
from .knn_model import KnnModel
from ..utils.data_store_utils import get_np_memmap, batch_cosine_similarity

class BaseTimeSeriesForecastingRunner(BaseRunner):
    """
    Runner for short term multivariate time series forecasting datasets.
    Typically, models predict the future 12 time steps based on historical time series.
    Features:
        - Evaluate at horizon 3, 6, 12, and overall.
        - Metrics: MAE, RMSE, MAPE. The best model is the one with the smallest mae at validation.
        - Loss: MAE (masked_mae). Allow customization.
        - Support curriculum learning.
        - Users only need to implement the `forward` function.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.dataset_name = cfg["DATASET_NAME"]
        # different datasets have different null_values, e.g., 0.0 or np.nan.
        self.null_val = cfg["TRAIN"].get("NULL_VAL", np.nan)    # consist with metric functions
        self.dataset_type = cfg["DATASET_TYPE"]
        self.exp_name = cfg["EXP_NAME"]

        # read scaler for re-normalization
        self.scaler = load_pkl("datasets/" + self.dataset_name + "/scaler_in{0}_out{1}.pkl".format(
                                                cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"]))
        # define loss
        self.loss = cfg["TRAIN"]["LOSS"]
        # define metric
        self.metrics = {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape}
        # curriculum learning for output. Note that this is different from the CL in Seq2Seq archs.
        self.cl_param = cfg.TRAIN.get("CL", None)
        if self.cl_param is not None:
            self.warm_up_epochs = cfg.TRAIN.CL.get("WARM_EPOCHS", 0)
            self.cl_epochs = cfg.TRAIN.CL.get("CL_EPOCHS")
            self.prediction_length = cfg.TRAIN.CL.get("PREDICTION_LENGTH")
            self.cl_step_size = cfg.TRAIN.CL.get("STEP_SIZE", 1)
        # evaluation horizon
        self.evaluation_horizons = [_ - 1 for _ in cfg["TEST"].get("EVALUATION_HORIZONS", range(1, 13))]
        assert min(self.evaluation_horizons) >= 0, "The horizon should start counting from 0."

    def init_training(self, cfg: dict):
        """Initialize training.

        Including loss, training meters, etc.

        Args:
            cfg (dict): config
        """

        super().init_training(cfg)
        for key, _ in self.metrics.items():
            self.register_epoch_meter("train_"+key, "train", "{:.4f}")

    def init_validation(self, cfg: dict):
        """Initialize validation.

        Including validation meters, etc.

        Args:
            cfg (dict): config
        """

        super().init_validation(cfg)
        for key, _ in self.metrics.items():
            self.register_epoch_meter("val_"+key, "val", "{:.4f}")

    def init_test(self, cfg: dict):
        """Initialize test.

        Including test meters, etc.

        Args:
            cfg (dict): config
        """

        super().init_test(cfg)
        for key, _ in self.metrics.items():
            self.register_epoch_meter("test_"+key, "test", "{:.4f}")

    def build_train_dataset(self, cfg: dict):
        """Build MNIST train dataset

        Args:
            cfg (dict): config

        Returns:
            train dataset (Dataset)
        """

        data_file_path = "{0}/data_in{1}_out{2}.pkl".format(cfg["TRAIN"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])
        index_file_path = "{0}/index_in{1}_out{2}.pkl".format(cfg["TRAIN"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])

        # build dataset args
        dataset_args = cfg.get("DATASET_ARGS", {})
        # three necessary arguments, data file path, corresponding index file path, and mode (train, valid, or test)
        dataset_args["data_file_path"] = data_file_path
        dataset_args["index_file_path"] = index_file_path
        dataset_args["mode"] = "train"

        dataset = cfg["DATASET_CLS"](**dataset_args)
        print("train len: {0}".format(len(dataset)))

        batch_size = cfg["TRAIN"]["DATA"]["BATCH_SIZE"]
        self.iter_per_epoch = math.ceil(len(dataset) / batch_size)

        return dataset

    @staticmethod
    def build_val_dataset(cfg: dict):
        """Build MNIST val dataset

        Args:
            cfg (dict): config

        Returns:
            validation dataset (Dataset)
        """
        data_file_path = "{0}/data_in{1}_out{2}.pkl".format(cfg["VAL"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])
        index_file_path = "{0}/index_in{1}_out{2}.pkl".format(cfg["VAL"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])

        # build dataset args
        dataset_args = cfg.get("DATASET_ARGS", {})
        # three necessary arguments, data file path, corresponding index file path, and mode (train, valid, or test)
        dataset_args["data_file_path"] = data_file_path
        dataset_args["index_file_path"] = index_file_path
        dataset_args["mode"] = "valid"

        dataset = cfg["DATASET_CLS"](**dataset_args)
        print("val len: {0}".format(len(dataset)))

        return dataset

    @staticmethod
    def build_test_dataset(cfg: dict):
        """Build MNIST val dataset

        Args:
            cfg (dict): config

        Returns:
            train dataset (Dataset)
        """

        data_file_path = "{0}/data_in{1}_out{2}.pkl".format(cfg["TEST"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])
        index_file_path = "{0}/index_in{1}_out{2}.pkl".format(cfg["TEST"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])

        # build dataset args
        dataset_args = cfg.get("DATASET_ARGS", {})
        # three necessary arguments, data file path, corresponding index file path, and mode (train, valid, or test)
        dataset_args["data_file_path"] = data_file_path
        dataset_args["index_file_path"] = index_file_path
        dataset_args["mode"] = "test"

        dataset = cfg["DATASET_CLS"](**dataset_args)
        print("test len: {0}".format(len(dataset)))

        return dataset

    def curriculum_learning(self, epoch: int = None) -> int:
        """Calculate task level in curriculum learning.

        Args:
            epoch (int, optional): current epoch if in training process, else None. Defaults to None.

        Returns:
            int: task level
        """

        if epoch is None:
            return self.prediction_length
        epoch -= 1
        # generate curriculum length
        if epoch < self.warm_up_epochs:
            # still warm up
            cl_length = self.prediction_length
        else:
            _ = ((epoch - self.warm_up_epochs) // self.cl_epochs + 1) * self.cl_step_size
            cl_length = min(_, self.prediction_length)
        return cl_length

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        """Feed forward process for train, val, and test. Note that the outputs are NOT re-scaled.

        Args:
            data (tuple): data (future data, history data). [B, L, N, C] for each of them
            epoch (int, optional): epoch number. Defaults to None.
            iter_num (int, optional): iteration number. Defaults to None.
            train (bool, optional): if in the training process. Defaults to True.

        Returns:
            tuple: (prediction, real_value). [B, L, N, C] for each of them.
        """

        raise NotImplementedError()

    def metric_forward(self, metric_func, args):
        """Computing metrics.

        Args:
            metric_func (function, functools.partial): metric function.
            args (list): arguments for metrics computation.
        """

        if isinstance(metric_func, functools.partial) and list(metric_func.keywords.keys()) == ["null_val"]:
            # support partial(metric_func, null_val = something)
            metric_item = metric_func(*args)
        elif callable(metric_func):
            # is a function
            metric_item = metric_func(*args, null_val=self.null_val)
        else:
            raise TypeError("Unknown metric type: {0}".format(type(metric_func)))
        return metric_item

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
        forward_return = list(self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True))
        # re-scale data
        prediction_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[0], **self.scaler["args"])
        real_value_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[1], **self.scaler["args"])
        # loss
        if self.cl_param:
            cl_length = self.curriculum_learning(epoch=epoch)
            forward_return[0] = prediction_rescaled[:, :cl_length, :, :]
            forward_return[1] = real_value_rescaled[:, :cl_length, :, :]
        else:
            forward_return[0] = prediction_rescaled
            forward_return[1] = real_value_rescaled
        if len(forward_return) == 6: # for graph learning
            loss = self.metric_forward(self.loss, forward_return[:5])
        else:
            loss = self.metric_forward(self.loss, forward_return[:2])
        # metrics
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return[:2])
            self.update_epoch_meter("train_"+metric_name, metric_item.item())
        return loss

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):
        """Validation details.

        Args:
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader
            train_epoch (int): current epoch if in training process. Else None.
            iter_index (int): current iter.
        """

        forward_return = self.forward(data=data, epoch=None, iter_num=None, train=False)
        # re-scale data
        prediction_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[0], **self.scaler["args"])
        real_value_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[1], **self.scaler["args"])
        # metrics
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, [prediction_rescaled, real_value_rescaled])
            self.update_epoch_meter("val_"+metric_name, metric_item.item())

    @torch.no_grad()
    @master_only
    def test(self, plot=False):
        """Evaluate the model.

        Args:
            train_epoch (int, optional): current epoch if in training process.
        """

        # test loop
        prediction = []
        real_value = []
        for _, data in tqdm(enumerate(self.test_data_loader), desc="testing"):
            forward_return = self.forward(data, epoch=None, iter_num=None, train=False)
            prediction.append(forward_return[0].detach().cpu())        # preds = forward_return[0]
            real_value.append(forward_return[1].detach().cpu())        # testy = forward_return[1]
        prediction = torch.cat(prediction, dim=0)
        real_value = torch.cat(real_value, dim=0)
        # re-scale data
        prediction = SCALER_REGISTRY.get(self.scaler["func"])(
            prediction, **self.scaler["args"])
        real_value = SCALER_REGISTRY.get(self.scaler["func"])(
            real_value, **self.scaler["args"])
        # summarize the results.
        # test performance of different horizon
        for i in self.evaluation_horizons:
            # For horizon i, only calculate the metrics **at that time** slice here.
            try:
                pred = prediction[:, i, :, :]
                real = real_value[:, i, :, :]
            except:
                pred = prediction[:, i, :]
                real = real_value[:, i, :]
            # metrics
            metric_results = {}
            for metric_name, metric_func in self.metrics.items():
                metric_item = self.metric_forward(metric_func, [pred, real])
                metric_results[metric_name] = metric_item.item()
            log = "Evaluate best model on test data for horizon " + \
                "{:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}"
            log = log.format(
                i+1, metric_results["MAE"], metric_results["RMSE"], metric_results["MAPE"])
            self.logger.info(log)
        # test performance overall
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, [prediction, real_value])
            self.update_epoch_meter("test_"+metric_name, metric_item.item())
            metric_results[metric_name] = metric_item.item()
        if plot:
            self.plot_data(prediction, real_value)


    @master_only
    def on_validating_end(self, train_epoch: Optional[int]):
        """Callback at the end of validating.

        Args:
            train_epoch (Optional[int]): current epoch if in training process.
        """

        if train_epoch is not None:
            self.save_best_model(train_epoch, "val_MAE", greater_best=False)

    @torch.no_grad()
    @master_only
    def create_data_store(self, cfg, output_dir="./data_store", subset="train", encoding_hidden_dim=512):
        """Evaluate the model.

        Args:
            train_epoch (int, optional): current epoch if in training process.
        """
        if subset == "train":
            cfg["TRAIN"]["DATA"]["SHUFFLE"] = False
            data_loader = self.build_train_data_loader(cfg)
        else:
            data_loader = self.build_val_data_loader(cfg)
        dstore_size = len(data_loader) * data_loader.batch_size * data_loader.dataset.data.size()[1]
        start_time = time.time()
        self.model.eval()


        import os
        if not os.path.exists(output_dir):
            print("{} does not exist. Make It.".format(output_dir))
            os.mkdir(output_dir)

        pred_len = cfg["DATASET_INPUT_LEN"]
        label_len = cfg["DATASET_OUTPUT_LEN"]
        encoding_hidden_dim = encoding_hidden_dim # TODO: this need to be unified
        info = {
            "pred_len": pred_len,
            "label_len": label_len,
            "encoding_hidden_dim": encoding_hidden_dim,
            "dstore_size": dstore_size,
            "node_num": data_loader.dataset.data.size()[1]
        }
        with open(os.path.join(output_dir, "info.json"), "w") as f:
            json.dump(info, f)

        predictions_np = get_np_memmap(output_dir, "{}_predictions.npy".format(subset), mode="w+", shape=(info["dstore_size"], info["pred_len"]))
        real_values_np = get_np_memmap(output_dir, "{}_real_values.npy".format(subset), mode="w+", shape=(info["dstore_size"], info["label_len"]))
        encoding_hiddens_np = get_np_memmap(output_dir, "{}_hiddens.npy".format(subset), mode="w+", shape=(info["dstore_size"], info["encoding_hidden_dim"]))
        data_indices_np = get_np_memmap(output_dir, "{}_data_indices.npy".format(subset), mode="w+", shape=(info["dstore_size"]))
        node_indices_np = get_np_memmap(output_dir, "{}_node_indices.npy".format(subset), mode="w+", shape=(info["dstore_size"]))

        start = 0
        for _, data in tqdm(enumerate(data_loader), desc="Creating Data Store from {}.".format(subset)):
            forward_return = self.forward(data, epoch=None, iter_num=None, train=False)
            prediction = forward_return[0]
            prediction = SCALER_REGISTRY.get(self.scaler["func"])(prediction, **self.scaler["args"])
            prediction = prediction.squeeze(-1).permute(0, 2, 1)
            real_value = forward_return[1]
            real_value = SCALER_REGISTRY.get(self.scaler["func"])(real_value, **self.scaler["args"])
            real_value = real_value.squeeze(-1).permute(0, 2, 1)

            B, N, _= prediction.size()
            assert len(forward_return) > 2, "forward_return must contains pred, real, hiddens"
            encoding_hidden = forward_return[-1]
            assert len(data) == 4, "future data, history data, idx, long history"
            data_indice = data[2][1].unsqueeze(-1)
            data_indice = data_indice.expand(B, N).contiguous()
            node_indice = torch.arange(N).unsqueeze(0).expand(B, N)

            prediction = prediction.contiguous().view(-1, prediction.size()[-1])
            real_value = real_value.contiguous().view(-1, real_value.size()[-1])
            encoding_hidden = encoding_hidden.contiguous().view(-1, encoding_hidden.size()[-1])
            data_indice = data_indice.contiguous().view(-1)
            node_indice = node_indice.contiguous().view(-1)

            end = start+B*N
            predictions_np[start:end, :] = prediction.cpu().detach().numpy()
            real_values_np[start:end, :] = real_value.cpu().detach().numpy()
            encoding_hiddens_np[start:end, :] = encoding_hidden.cpu().detach().numpy()
            data_indices_np[start:end] = data_indice.cpu().detach().numpy()
            node_indices_np[start:end] = node_indice.cpu().detach().numpy()

            start = end
        end_time = time.time()
        print("Total inference time: {} minutes.".format((end_time - start_time) / 60))
        # test read np memmap

    @torch.no_grad()
    @master_only
    def test_knn_process(self, cfg, dstore_dir, used_hidden="hiddens", k=100, metric="l2", knn_weight=0.5, train_epoch: int = None):
        """The whole test process.

        Args:
            cfg (dict, optional): config
            train_epoch (int, optional): current epoch if in training process.
        """

        # init test if not in training process
        if train_epoch is None:
            self.init_test(cfg)

        self.on_test_start()

        test_start_time = time.time()
        self.model.eval()

        # test
        self.test_knn(cfg=cfg, dstore_dir=dstore_dir, used_hidden=used_hidden, k=k, metric=metric, knn_weight=knn_weight)

        test_end_time = time.time()
        self.update_epoch_meter("test_time", test_end_time - test_start_time)
        # print test meters
        self.print_epoch_meters("test")
        if train_epoch is not None:
            # tensorboard plt meters
            self.plt_epoch_meters("test", train_epoch // self.test_interval)

        self.on_test_end()


    @torch.no_grad()
    @master_only
    def test_knn(self, cfg, dstore_dir, used_hidden="hiddens", k=100, metric="l2", knn_weight=1, t=0.1):
        """Evaluate the model.

        Args:
            train_epoch (int, optional): current epoch if in training process.
        """
        knn_model = KnnModel(dstore_dir=dstore_dir, used_hidden=used_hidden, k=k, metric=metric)
        # test loop
        predictions = []
        real_values = []
        data_loader = self.test_data_loader
        # data_loader =  self.build_train_data_loader(cfg)
        knn_better_stds = []
        model_better_stds = []

        prediction_worses = []
        prediction_better = []
        prediction_knn_better = []
        prediction_worses_labels = []
        prediction_better_labels = []


        for _, data in tqdm(enumerate(data_loader), desc="Testing"):
            forward_return = self.forward(data, epoch=None, iter_num=None, train=False)
            prediction = forward_return[0]
            prediction = SCALER_REGISTRY.get(self.scaler["func"])(prediction, **self.scaler["args"])
            prediction = prediction.squeeze(-1).permute(0, 2, 1)
            prediction = prediction.cpu()
            real_value = forward_return[1]
            real_value = SCALER_REGISTRY.get(self.scaler["func"])(real_value, **self.scaler["args"])
            real_value = real_value.squeeze(-1).permute(0, 2, 1)
            real_value = real_value.cpu()

            B, N, _= prediction.size()
            hiddens = forward_return[-1]
            data_indice = data[2][1].unsqueeze(-1)
            data_indice = data_indice.expand(B, N).contiguous()
            node_indice = torch.arange(N).unsqueeze(0).expand(B, N)

            queries = hiddens
            hiddens_sim = batch_cosine_similarity(hiddens, hiddens) # B, N, N
            topk, topk_indices = torch.topk(hiddens_sim, k=5, dim=-1) # B, N, k

            queries = queries.contiguous().view(B*N, -1)
            knn_vals, dists, knns, probs = knn_model.get_knn_prob(queries=queries, k=k, t=t)
            knn_vals = knn_vals.contiguous().view(B, N, knn_vals.size()[1], -1).to(prediction.device)

            knn_nodes = torch.from_numpy(knn_model.nodes[knns]).view(B, N, k) #B, N, topk
            knn_indices = torch.from_numpy(knn_model.data_indices[knns]).view(B, N, k)

            probs = probs.to(prediction.device)
            probs = probs.contiguous().view(B, N, probs.size()[-1])
            knn_vals_sum = knn_vals * probs.unsqueeze(-1)
            knn_vals_sum = torch.sum(knn_vals_sum, dim=2)
            prediction_knn = knn_vals_sum * knn_weight + prediction * (1 - knn_weight)

            # prediction_knn = knn_vals_sum

            # TODO for plot, this should be remove!!!
            if self.metrics["MAE"](prediction_knn[:, :, 2], real_value[:, :, 2], self.null_val) < self.metrics["MAE"](prediction[:, :, 2], real_value[:, :, 2], self.null_val) - 0.1: #and \
            #    self.metrics["MAE"](prediction_knn[:, :, 5], real_value[:, :, 5], self.null_val) < self.metrics["MAE"](prediction[:, :, 5], real_value[:, :, 5], self.null_val) - 0.1 and \
            #    self.metrics["MAE"](prediction_knn[:, :, 8], real_value[:, :, 8], self.null_val) < self.metrics["MAE"](prediction[:, :, 8], real_value[:, :, 8], self.null_val) - 0.2 and \
            #    self.metrics["MAE"](prediction_knn[:, :, 11], real_value[:, :, 11], self.null_val) < self.metrics["MAE"](prediction[:, :, 11], real_value[:, :, 11], self.null_val) - 0.2:

                print("knn mae: {}, prediction mae: {}".format(self.metrics["MAE"](prediction_knn[:, :, 2], real_value[:, :, 2]),
                                                               self.metrics["MAE"](prediction[:, :, 2], real_value[:, :, 2])))
                # print("knn is better in this batch, batch size: {}".format(B))
                # print("knn node: {}".format(knn_nodes[0, 0, :]))
                # print("knn indices: {}".format(knn_indices[0, 0, :]))
                print("data indices: {}".format(data[3]))
                knn_better_stds.append(torch.std_mean(SCALER_REGISTRY.get(self.scaler["func"])(data[1], **self.scaler["args"])))
                prediction_worses.append(prediction)
                prediction_knn_better.append(prediction_knn)
                prediction_worses_labels.append(real_value)
            else:
                model_better_stds.append(torch.std_mean(SCALER_REGISTRY.get(self.scaler["func"])(data[1], **self.scaler["args"])))
                prediction_better.append(prediction)
                prediction_better_labels.append(real_value)
            # prediction = knn_vals_sum * knn_weight + (1 - knn_weight) * prediction

            prediction = prediction_knn
            predictions.append(prediction.permute(0, 2, 1).unsqueeze(-1))
            real_values.append(real_value.permute(0, 2, 1).unsqueeze(-1))



        predictions = torch.cat(predictions, dim=0)
        real_values = torch.cat(real_values, dim=0)

        # summarize the results.
        # test performance of different horizon
        for i in self.evaluation_horizons:
            # For horizon i, only calculate the metrics **at that time** slice here.
            pred = predictions[:, i, :, :]
            real = real_values[:, i, :, :]
            # metrics
            metric_results = {}
            for metric_name, metric_func in self.metrics.items():
                metric_item = self.metric_forward(metric_func, [pred, real])
                metric_results[metric_name] = metric_item.item()
            log = "Evaluate best model on test data for horizon " + \
                "{:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}"
            log = log.format(
                i+1, metric_results["MAE"], metric_results["RMSE"], metric_results["MAPE"])
            self.logger.info(log)
        # test performance overall
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, [predictions, real_values])
            self.update_epoch_meter("test_"+metric_name, metric_item.item())
            metric_results[metric_name] = metric_item.item()

    @torch.no_grad()
    @master_only
    def plot_result(self, plot=False, eval_len=20):
        """Evaluate the model.

        Args:
            train_epoch (int, optional): current epoch if in training process.
        """

        # test loop
        prediction = []
        real_value = []
        for _, data in tqdm(enumerate(self.test_data_loader), desc="testing"):
            forward_return = self.forward(data, epoch=None, iter_num=None, train=False)
            prediction.append(forward_return[0].detach().cpu())        # preds = forward_return[0]
            real_value.append(forward_return[1].detach().cpu())        # testy = forward_return[1]
            # if _ > 2:
            #     break

        prediction = torch.cat(prediction, dim=0)
        real_value = torch.cat(real_value, dim=0)
        # re-scale data
        prediction = SCALER_REGISTRY.get(self.scaler["func"])(
            prediction, **self.scaler["args"])
        real_value = SCALER_REGISTRY.get(self.scaler["func"])(
            real_value, **self.scaler["args"])

        #draw 1 horizon for all prediction as figure
        if plot:
            self.plot_data(prediction, real_value)

        # todo : summarize the results. eval metrics on eval_len
        # test performance of different eval length on 12 horizon
        target_node = 235
        metric_results = []
        for i in range(prediction.shape[0]):
            pred = prediction[i, :, target_node]
            real = real_value[i, :, target_node]           
            for metric_name, metric_func in self.metrics.items():
                if metric_name == "MAE":
                    metric_item = self.metric_forward(metric_func, [pred, real])
                    metric_results.append(metric_item.item())
                else:
                    pass              
        out_put = {}
        out_put['metric_results'] = metric_results
        out_put['prediction'] = prediction[:,:]
        out_put['real'] = real_value[:,:]        
        out_put['exp_name'] = self.exp_name
        out_put['target_node'] = target_node


        with open("ablation" + "/{0}_prediction.pkl".format(self.exp_name), "wb") as f:
                pickle.dump(out_put, f)

