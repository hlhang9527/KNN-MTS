from ast import parse
from operator import index
import os
from random import sample, seed
import faiss
import math
from basicts.runners.data_store import DataStore
import numpy as np
import time
import argparse


class IndexBuilder:

  def __init__(self, dstore_dir, used_hidden="hiddens", use_gpu=False, metric="l2", suffix=""):

      self.dstore_dir = dstore_dir
      self.dstore = DataStore.from_pretrained(dstore_dir=dstore_dir, used_hidden=used_hidden, mode="r", subset="train")
      self.use_gpu = use_gpu
      self.metric = metric
      self.suffix =suffix
      self.used_hidden = used_hidden

  def exists(self):
    return os.path.exists(self.trained_file) and os.path.exists(self.faiss_file)

  @property
  def trained_file(self):
    file_path = os.path.join(self.dstore_dir, "faiss_store.trained.{}.{}{}".format(self.used_hidden, self.metric, self.suffix))
    return file_path

  @property
  def faiss_file(self):
    file_path = os.path.join(self.dstore_dir, "faiss_store.{}.{}{}".format(self.used_hidden, self.metric, self.suffix))
    return file_path

  def get_auto_index_type(self):
    """we choose index type by https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index"""
    dstore_size = self.dstore.dstore_size
    if dstore_size < 3000:
        return "IDMap,,Flat"
    clusters = min(int(4 * math.sqrt(dstore_size)), dstore_size // 30, 131072)
    if dstore_size < 30000:
        return "IDMap,,Flat"
    if dstore_size < 10 ** 6:
        return f"OPQ32_{self.dstore.hidden_size},IVF{clusters},PQ32"  # we use 64 here since faiss does not support >64 in gpu mode
    return f"OPQ32_{self.dstore.hidden_size},IVF{clusters}_HNSW32,PQ32"

  def build(self, index_type, chunk_size=1000000, seed=None, start=0, overwrite=False):
    if index_type == "auto":
      index_type = self.get_auto_index_type()

    self.train(index_type=index_type, max_num=chunk_size, seed=seed, overwrite=overwrite)
    print("Adding Keys.")
    pretrained_file = self.trained_file

    if os.path.exists(self.faiss_file) and not overwrite:
      pretrained_file = self.faiss_file
      print("faiss index file exists, use it as pretrain index")

    index = faiss.read_index(pretrained_file)
    if pretrained_file == self.faiss_file:
      start = index.ntotal
    print("start from {} lines, due to pretrained faiss file: {}".format(start, self.faiss_file))
    dstore_size = self.dstore.dstore_size
    start_time = time.time()
    while start < dstore_size:
      end = min(dstore_size, start + chunk_size)
      to_add = np.array(self.dstore.keys[start:end])
      if self.metric == "cosine":
        norm = np.sqrt(np.sum(to_add ** 2, axis=-1, keepdims=True))
        if (norm == 0).any():
            print(f"found zero norm vector in {self.dstore.dstore_dir}")
            norm = norm + 1e-10
        to_add = to_add / norm
      index.add(to_add.astype(np.float32))
      start = end

      print("Add {} tokens so far".format(index.ntotal))
      faiss.write_index(index, self.faiss_file)
    print("Adding total {} keys.".format(index.ntotal))
    print("Adding took {} s".format(time.time() - start_time))
    print("Writing Index")
    start = time.time()
    faiss.write_index(index, self.faiss_file)
    print("Writing index took {} s".format(time.time() - start_time))
    print("Wrote data to {}".format(self.faiss_file))


  def train(self, index_type, max_num=None, seed=None, overwrite=False):
    hidden_size, dstore_size = self.dstore.hidden_size, self.dstore.dstore_size
    if os.path.exists(self.trained_file) and not overwrite:
      print("trained file already existes. Us existing file: {}".format(self.trained_file))
      return

    metric = faiss.METRIC_L2 if self.metric == "l2" else  faiss.METRIC_INNER_PRODUCT
    index = faiss.index_factory(hidden_size, index_type, metric)

    if self.use_gpu:
      print("Using gpu for training")
      res = faiss.StandardGpuResources()
      co = faiss.GpuClonerOptions()

      index = faiss.index_cpu_to_gpu(res, 0, index, co)

    if self.dstore.dstore_size < max_num:
      sample_keys = np.array(self.dstore.keys.astype(np.float32))
    else:
      np.random.seed(seed)
      max_num = max_num or self.dstore.dstore_size
      sample_keys = np.array(self.dstore.keys[-max_num:].astype(np.float32))
      if self.metric == "cosine":
        norm = np.sqrt(np.sum(sample_keys ** 2, axis=-1, keepdims=True))
        if (norm == 0).any():
          print("find zero norm vector in {}".format(self.dstore.dstore_dir))
          norm = norm + 1e-10
        sample_keys = sample_keys / norm

    start = time.time()
    print("Training Index")
    index.verbose = True
    index.train(sample_keys)
    print("Training took {} s".format(time.time() - start))
    if self.use_gpu:
      index = faiss.index_gpu_to_cpu(index)
    print("Writing index after training")

    start = time.time()
    faiss.write_index(index, self.trained_file)
    print("Writing index took {} s".format(time.time() - start))


def build(dstore_dir, used_hidden="tsformer", index_type="auto", use_gpu=False, metric="l2", suffix="", overwrite=False, seed=None, chunk_size=1000000):
  index_buider = IndexBuilder(dstore_dir=dstore_dir, use_gpu=use_gpu, metric=metric, suffix=suffix)
  if overwrite or not index_buider.exists():
    index_buider.build(index_type=index_type, seed=seed, chunk_size=chunk_size, overwrite=overwrite)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--index_type", default="auto", type=str)
  parser.add_argument("--use_gpu", default=True, action='store_true')
  parser.add_argument('--metric', type=str, default="l2", choices=["l2", "ip", "cosine"],
                        help='faiss index metric, l2 for L2 distance, ip for inner product, '
                        'cosine for cosine similarity')
  parser.add_argument("--suffix", default="", type=str)
  parser.add_argument('--chunk_size', default=1000000, type=int,
                        help='can only load a certain amount of data to memory at a time.')
  parser.add_argument('--seed', type=int, default=123, help='random seed')
  parser.add_argument("--overwrite", action="store_true", default=False,
                        help="if True, delete old faiss_store files before generating new ones")
  parser.add_argument("--dstore_dir", type=str, default="./data_store", help="paths to data store. if provided multiple,"
                                                                      "use ',' as separator")
  parser.add_argument("--used_hidden", default="tsformer", type=str)
  args = parser.parse_args()
  build(args.dstore_dir, used_hidden=args.used_hidden, index_type=args.index_type, use_gpu=args.use_gpu, seed=args.seed,
        metric=args.metric, suffix=args.suffix, overwrite=args.overwrite, chunk_size=args.chunk_size)



