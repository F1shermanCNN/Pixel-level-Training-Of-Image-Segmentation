# Pixel-level-Training-Of-Image-Segmentation
计算机视觉 HW2 Task3
首先按照requirements.txt配置环境
```bash
pip install -r requirements.txt
```
# 1.训练测试
```bash
python train.py --epochs 50 --batch_size 16 --image_size 256 --lr 0.0005 --save_dir checkpoints/testrun
```
训练参数均可更改

# 2.评估测试
首先更改evaluate.py中的experiments参数，将其更改为所下载的参数的地址，随后运行
```bash
python evaluate.py
```
# 3.混淆矩阵和错例
首先更改confusion_matrix.py，main中的WEIGHT_PATH等参数，随后运行
```bash
python confusion_matrix.py
```
