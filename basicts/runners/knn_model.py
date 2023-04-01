from curses import noecho
from dis import dis
import os
import time

import faiss
import numpy as np
import torch

from torch.nn import functional as F
from basicts.runners.data_store import DataStore

class KnnModel(object):

  def __init__(self, dstore_dir, used_hidden="tsformer", probe=32,
              sim_func="", k=10, cuda=-1, efsearch=8, metric="l2", suffix=""):

    self.index_file = os.path.join(dstore_dir, "faiss_store.{}.{}{}".format(used_hidden, metric, suffix))
    self.dstore_dir = dstore_dir
    self.probe = probe
    self.efsearch = efsearch
    print("Knn Model index file is: {}, topk: {}, used_hidden: {}, metric: {}".format(self.index_file, k, used_hidden, metric))
    t = time.time()
    self.data_store = DataStore.from_pretrained(dstore_dir=dstore_dir, used_hidden=used_hidden, mode="r", subset="train")
    print("Reading data store took: {}".format(time.time() - t))

    self.dstore_size = self.data_store.dstore_size
    self.hidden_size = self.data_store.hidden_size

    self.vals = self.data_store.vals
    self.keys = self.data_store.keys
    self.nodes = self.data_store.node_indices_np
    self.data_indices = self.data_store.data_indices_np

    self.used_hidden = used_hidden

    self.k = k
    self.metric = metric
    self.sim_func = sim_func

    self.index = self.setup_faiss()

  def setup_faiss(self):
    """setup faiss index"""
    if not os.path.exists(self.dstore_dir):
      raise ValueError(f'Dstore path not found: {self.dstore_dir}')

    start = time.time()
    print(f'Reading faiss index, with nprobe={self.probe},  efSearch={self.efsearch} ...')
    index = faiss.read_index(self.index_file, faiss.IO_FLAG_ONDISK_SAME_DIR)
    print(f'Reading faiss of size {index.ntotal} index took {time.time() - start} s')
    try:
      faiss.ParameterSpace().set_index_parameter(index, "nprobe", self.probe)
      faiss.ParameterSpace().set_index_parameter(index, "quantizer_efSearch", self.efsearch)
    except Exception as e:
      print(f"faiss index {self.index_file} does not have parameter nprobe or efSearch")
    return index

  def get_knns(self, queries, k=0):
    """
    get distances and knns from queries
    Args:
        queries: Tensor of shape [num, hidden]
        k: number of k, default value is self.k
    Returns:
        dists: knn dists. np.array of shape [num, k]
        knns: knn ids. np.array of shape [num, k]
    """
    k = k or self.k
    if isinstance(queries, torch.Tensor):
      queries = queries.detach().cpu().float().data.numpy()
    dists, knns = self.index.search(queries.astype(np.float32), k=k)
    return dists, knns

  def get_knn_prob(self, queries, k=0, return_knn=False, t=1.0):
    """
    Args:
        queries: Tensor of shape [batch, hidden]
        output_size: int
        k: int, number of neighbors
        return_knn: if True, return the knn dists and knn vals
        t: temperature
    Returns:
        probs: tensor of shape [batch, output_size]
        knn dists: np.array of shape [batch, K]
        knn keys: np.array of shape [batch, K]
    """
    k = k or self.k

    def dist_func(dists, knns, queries, function=None):
        """
        计算L2 distance
        Args:
            dists: knn distances, [batch, k]
            knns: knn ids, [batch, k]
            queries: qeuries, [batch, hidden]
            function: sim function
            k: number of neighbors
        Returns:
            dists. tensor of shape [batch, k]
        """
        if not function:
            # Default behavior for L2 metric is to recompute distances.
            # Default behavior for IP metric is to return faiss distances.
            if isinstance(queries, torch.Tensor):
              qsize = queries.size()
            else:
              queries = torch.tensor(queries)
              qsize = queries.shape
            if self.metric == 'l2':
                # [batch, k, hidden]
                try:
                  knns_vecs = torch.from_numpy(self.keys[knns]).to(queries.device) # if gpu is small, use cpu
                  # [batch, k, hidden]
                  query_vecs = queries.view(qsize[0], 1, qsize[1]).repeat(1, k, 1)
                  l2 = torch.sum((query_vecs - knns_vecs) ** 2, dim=2)
                  return -1 * l2
                except:
                  print("WARNING: You are using cpu to compute!")
                  queries_cpu = queries.cpu()
                  knns_vecs = torch.from_numpy(self.keys[knns]).to(queries_cpu.device) # if gpu is small, use cpu
                  # [batch, k, hidden]
                  query_vecs = queries.view(qsize[0], 1, qsize[1]).repeat(1, k, 1).to(queries_cpu.device)
                  l2 = torch.sum((query_vecs - knns_vecs) ** 2, dim=2).to(queries.device)

            return dists

        if function == 'dot':
            qsize = queries.size()
            keys = torch.from_numpy(self.keys[knns])  # .cuda()
            return (keys * queries.view(qsize[0], 1, qsize[1])).sum(dim=-1)

        if function == 'do_not_recomp_l2':
            return -1 * dists

        raise ValueError("Invalid knn similarity function!")

    # [batch, k]; [batch, k]
    dists, knns = self.get_knns(queries, k=k)
    knns = torch.from_numpy(knns)
    dists = torch.from_numpy(dists)  ##.cuda()
    # [batch, k]
    dists = dist_func(dists, knns, queries, function=self.sim_func)
    assert len(dists.size()) == 2
    # [batch, k]
    probs = F.softmax(dists / t, dim=-1)
    # [batch, k]
    knn_vals = torch.from_numpy(self.vals[knns])  ##.cuda()
    return knn_vals, dists, knns, probs

if __name__ == "__main__":
  knn_model = KnnModel(dstore_dir="./data_store", k=10)
  query = np.random.random((2, 96))
  knn_vals, dists, knns, probs = knn_model.get_knn_prob(queries=query, k=10)
  # return knn_vals, probs, is tensor
  print("knn vals size: {}".format(knn_vals.size()))


