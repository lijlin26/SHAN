# Simplicial Hyperbolic Attention Network
This is the source code of paper ["Multi-Order Relations Hyperbolic Fusion for Heterogeneous  Graphs"](https://dl.acm.org/doi/10.1145/3583780.3614979)
![](fig/SAN_framework.png)
The source code is based on [HGCN](https://github.com/HazyResearch/hgcn)

## Get Started
### Requirements
- dgl==2.2.1
- torch==2.2.2
- torch_geometric==2.5.2
- torch-scatter==2.1.2
- torch-sparse==0.6.18
### Datasets
```bash
mkdir data
cd data
```
We use datasets (ACM, IMDB, DBLP) from [Graph Transformer Networks](https://github.com/seongjunyun/Graph_Transformer_Networks/tree/master). Download and extract [data.zip](https://drive.google.com/file/d/1Nx74tgz_-BDlqaFO75eQG6IkndzI92j4/view) into data folder.


### Run
- ACM
```bash
python main.py --dataset acm --decoder linear
```
- IMDB
```bash
python main.py --dataset imdb --decoder gat
```
- DBLP
```bash
python main.py --dataset dblp --decoder gat --K 1 --decoder_residual 0
```
## Reference
If this work is useful for your research, please cite our work:
```
@inproceedings{10.1145/3583780.3614979,
author = {Li, Junlin and Sun, Yueheng and Shao, Minglai},
title = {Multi-Order Relations Hyperbolic Fusion for Heterogeneous Graphs},
year = {2023},
booktitle = {Proceedings of the 32nd ACM International Conference on Information and Knowledge Management}
}
```