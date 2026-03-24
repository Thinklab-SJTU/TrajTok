<div align="center">   

# TrajTok: What makes for a good trajectory tokenizer in behavior generation?
</div>

>Official implementation of paper [TrajTok: What makes for a good trajectory tokenizer in behavior generation?](https://openreview.net/pdf?id=Zvy2agYouY). *Zhiyuan Zhang, Xiaosong Jia, Guanyu Chen, Qifeng Li, Zuxuan Wu, Yu-Gang Jiang, Junchi Yan*. **ICLR 2026**

**First Place of [Waymo Open Sim Agents Challenge 2025](https://waymo.com/open/challenges/) 🏆** 

![rank](./assets/rank.png)


![rank](./assets/sim.png)

## Fast WOSAC Metric 🚀

It is **very slow** to compute the WOSAC metrics with the official code. It usually takes 10-30 seconds to evaluate a single scenario and 100+ hours on the whole validation set. ([here are issues](https://github.com/waymo-research/waymo-open-dataset/issues/877))

For **fast evaluation and quick development**, we developed the **fast WOSAC metric** that reduces the time to about **0.4s/scenario** while maintain the error of each indicator from the official less than $10^{-6}$. The tool supports both **2024 and 2025 version** of WOSAC metric and is easy to use in both **online and offline** modes.

Please refer to() for more information.

## Environment Setup

```bash
conda create -y -n trajtok python=3.11.9
conda activate trajtok
conda install -y -c conda-forge ffmpeg=4.3.2
pip install -r requirements.txt
pip install torch_geometric
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install --no-deps waymo-open-dataset-tf-2-12-0==1.6.4
```

## Data Preparation

Step1. Download [Waymo Open Motion Dataset](https://waymo.com/open/download/) v1.3.0.
Step2. run data preprocess script.

```bash
# set INPUT_DIR as your dataset path (.../scenario/) before running
bash data_preprocess.sh
```

## Train

```bash
python run.py experiment=train task_name=train
```

## Evaluation

```bash
python run.py experiment=inference task_name=eval
```

## Tokenization

```bash
python -m src.smart.tokens.trajtok
```

## Citation

```
@inproceedings{zhang2026trajtok,
  title={TrajTok: What makes for a good trajectory tokenizer in behavior generation?},
  author={Zhiyuan Zhang and Xiaosong Jia and Guanyu Chen and Qifeng Li and Zuxuan Wu and Yu-Gang Jiang and Junchi Yan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## Acknowledgement

Thansk for these excellent opensource works and models: [SMART](https://github.com/rainmaker22/SMART) [CatK](https://github.com/NVlabs/catk).