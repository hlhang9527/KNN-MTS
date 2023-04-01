This repo contains all the baselines and our proposed framework code. 
To run the code:

First:
### Python

Python >= 3.6 (recommended >= 3.9).

[Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) are recommended to create a virtual python environment.



```bash
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

After ensuring that PyTorch is installed correctly, you can install other dependencies via:

```bash
pip install -r requirements.txt
```

Second:



- **Download Raw Data**

    You can download all the raw datasets at [Google Drive](https://drive.google.com/drive/folders/14EJVODCU48fGK0FkyeVom_9lETh80Yjp) and unzip them to `datasets/raw_data/`.

- **Pre-process Data**

    ```bash
    cd /path/to/your/project
    python scripts/data_preparation/${DATASET_NAME}/generate_training_data.py
    ```

    Replace `${DATASET_NAME}` with one of `PEMS-BAY`, `PEMS03`, `PEMS04`, `PEMS07`, `PEMS08`, or any other supported dataset. The processed data will be placed in `datasets/${DATASET_NAME}`.

    Or you can pre-process all datasets by.

    ```bash
    cd /path/to/your/project
    bash scripts/data_preparation/all.sh
    ```

- **Pre-train Model** 
    if you want to pretrain your own model, choose a base model path and using run.py to generate the model yourself. Or you can use our pretrained model in training_log as well.
```bash
python run.py -c examples/GWNet/GWNet_PEMS04.py --gpus '0'
```
    save the PATH

- **Using pretrained model in the framework and test the performance**
```bash
python test.py --cfg "PATH/TO/COFIG" --ckpt "PATH/TO/MODEL" --gpus "0" --task "create_data_store" --dstore_dir "./data_store/MODEL"   
python run_index_build.py --dstore_dir "./data_store/MODEL/" 
python test.py --cfg "PATH/TO/COFIG" --ckpt "PATH/TO/MODEL" --gpus "0" --task "knn_test" --dstore_dir "./data_store/MODEL"   
```

