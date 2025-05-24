# MiniMaxAD

This repository is the official implementation of [MiniMaxAD: A Lightweight Autoencoder for Feature-rich Anomaly Detection](https://arxiv.org/abs/2405.09933).

## Requirements

To install requirements:

```setup
conda create -n MMAD python=3.8.10
conda activate MMAD
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
## Prepare Datasets

### MVTec AD

Download the MVTec-AD dataset from [URL](https://www.mvtec.com/company/research/datasets/mvtec-ad). The MVTec-AD dataset directory should be as follows:
```
|-- path/to/your/data
    |-- mvtec_anomaly_detection
        |-- bottle
        |-- cable
        |-- capsule
        |-- ....
```


### VisA

Download the VisA dataset from [URL](https://github.com/amazon-science/spot-diff). Preprocess the dataset to `VisA_pytorch` in 1-class mode by their official scripts 
[code](https://github.com/amazon-science/spot-diff). `VisA_pytorch/` will be like:
```
|-- path/to/your/data
    |-- VisA_pytorch
        |-- 1cls
            |-- candle
            |-- capsules
            |-- cashew
            |-- ....
```

### GoodsAD

Download the GoodsAD dataset from [URL](https://github.com/jianzhang96/GoodsAD). The GoodsAD dataset directory should be as follows:
```
|-- path/to/your/data
    |-- GoodsAD
        |-- cigarette_box
        |-- drink_bottle
        |-- drink_can
        |-- ....
```

### AeBAD-S

Download the AeBAD-S dataset from [URL](https://github.com/zhangzilongc/MMR). The AeBAD_S dataset directory should be as follows:
```
|-- path/to/your/data
    |-- AeBAD
        |-- AeBAD_S
            |-- ground_truth
                |-- ablation
                |-- breakdown
                |-- ...
            |-- test
                |-- ablation
                |-- breakdown
                |-- ...
            |-- train
                |-- good
```

### Real-IAD
To download the RealIAD dataset, please contact the authors via [URL](https://realiad4ad.github.io/Real-IAD/). The RealIAD dataset directory should be as follows:
```
|-- path/to/your/data
    |-- Real-IAD
        |-- realiad_512
            |-- audiojack
            |-- bottle_cap
            |-- ....
        |-- realiad_jsons
            |-- realiad_jsons
            |-- realiad_jsons_fuiad_0.0
            |-- ...
```

## Training
Download the pretrained weights of UniRepLKNet from [URL](https://github.com/AILab-CVC/UniRepLKNet). Unzip the file to `./ckpts/`. Edit `./config.ini` to set the path and necessary parameters.
Run `train_single_*.py` to run single-class training. Use `train_uni_*.py` to run multi-class training.

## Citation
```
@misc{wang2024minimaxad,
      title={MiniMaxAD: A Lightweight Autoencoder for Feature-Rich Anomaly Detection}, 
      author={Fengjie Wang and Chengming Liu and Lei Shi and Pang Haibo},
      year={2024},
      eprint={2405.09933},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2405.09933}, 
}
```

## Acknowledgement
Our work is inspired by the following outstanding works: [RD](https://github.com/hq-deng/RD4AD),[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet),[ReContrast](https://github.com/guojiajeremy/ReContrast).