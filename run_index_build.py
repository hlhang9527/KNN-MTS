from basicts.runners.index_builder import IndexBuilder
import argparse

def build(dstore_dir, used_hidden="hiddens", index_type="auto", use_gpu=False, metric="l2", suffix="", overwrite=False, seed=None, chunk_size=1000000):
  index_buider = IndexBuilder(dstore_dir=dstore_dir, use_gpu=use_gpu, metric=metric, suffix=suffix, used_hidden=used_hidden)
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
  parser.add_argument("--used_hidden", default="hiddens", type=str)
  args = parser.parse_args()
  build(args.dstore_dir, used_hidden=args.used_hidden, index_type=args.index_type, use_gpu=args.use_gpu, seed=args.seed,
        metric=args.metric, suffix=args.suffix, overwrite=args.overwrite, chunk_size=args.chunk_size)