from curses import use_default_colors
from operator import mod
import os
from statistics import mode
import numpy as np
import time
import json
from ..utils.data_store_utils import get_np_memmap
import random


class DataStore(object):

  def __init__(self, dstore_dir, mode="r", subset="train", used_hidden="hiddens"):
    self.dstore_dir = dstore_dir
    self.info_dict_file = os.path.join(dstore_dir, "info.json")
    with open(self.info_dict_file, "r") as f:
      info = json.load(f)
    self.info_dict = info

    self.predictions_np = get_np_memmap(dstore_dir, "{}_predictions.npy".format(subset), mode=mode, shape=(info["dstore_size"], info["pred_len"]))
    self.real_values_np = get_np_memmap(dstore_dir, "{}_real_values.npy".format(subset), mode=mode, shape=(info["dstore_size"], info["label_len"]))
    self.hiddens_np = get_np_memmap(dstore_dir, "{}_{}.npy".format(subset, used_hidden), mode=mode, shape=(info["dstore_size"], info["encoding_hidden_dim"]))
    self.data_indices_np = get_np_memmap(dstore_dir, "{}_data_indices.npy".format(subset), mode=mode, shape=(info["dstore_size"]))
    self.node_indices_np = get_np_memmap(dstore_dir, "{}_node_indices.npy".format(subset), mode=mode, shape=(info["dstore_size"]))

    self.dstore_size = info["dstore_size"]
    self.hidden_size = self.info_dict["encoding_hidden_dim"]
    self.used_hidden = used_hidden
    print("used hidden is {}.".format(used_hidden))
    self.keys = self.hiddens_np
    self.vals = self.real_values_np

    self.print_one_line()

  @property
  def info(self):
    return self.info_dict

  def print_one_line(self):
    print("print one random line of dstore.")
    random_index = random.randint(0, self.info_dict["dstore_size"])
    print(self.predictions_np[random_index, :])
    print(self.real_values_np[random_index, :])
    print(self.hiddens_np[random_index, :])
    print(self.data_indices_np[random_index])
    print(self.node_indices_np[random_index])

  @classmethod
  def from_pretrained(cls, dstore_dir, used_hidden="hiddens", mode="r", subset="train"):
    return cls(dstore_dir, used_hidden=used_hidden, mode=mode, subset=subset)








