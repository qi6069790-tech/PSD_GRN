# PSD-GRN: Pseudotime-Enhanced Signed Directed Graph Learning for Gene Regulatory Network Inference
This project provides the implemenration of PSD-GRN and the experimental code for directed regulatory edge existence prediction and signed regulation prediction

## Environment Setup
- Python 3.9.25
- PyTorch 2.8.0 + CUDA 12.8
- PyTorch Geometric 2.6.1
- PyTorch Geometric Signed Directed 1.1.1
- DGL 2.2.1
- torch-scatter 2.1.2
- torch-sparse 0.6.18
- torch-spline-conv 1.2.2
- NumPy 2.0.2
- SciPy 1.13.1
- scikit-learn 1.6.1
- pandas 2.3.3
- NetworkX 3.2.1
- Matplotlib 3.9.4


## Running the Code
From the project root, enter the 'src' directory and run the following commands:
```
cd src
```


### 2C Task: Directed Regulatory Edge Existence Prediction
```
python train_psdgrn_2c.py --dataset 你的数据集 --expression_file 你的表达矩阵.csv --num_classes 2 --K 1 --q 0.1 --hidden 64 --lr 0.001 --weight_decay 0.05 --dropout 0.5 --epochs 1000 --patience 20 --checkpoint 25 --train_ratio 0.8 --val_ratio 0.1 --runs 5
```

### 3C Task: Signed Regulation Prediction
```
python train_psdgrn_3c.py --dataset 你的数据集 --expression_file 你的表达矩阵.csv --num_classes 3 --K 1 --q 0.1 --hidden 64 --lr 0.001 --weight_decay 0.05 --dropout 0.5 --epochs 1000 --patience 20 --checkpoint 25 --train_ratio 0.8 --val_ratio 0.1 --runs 5
```
