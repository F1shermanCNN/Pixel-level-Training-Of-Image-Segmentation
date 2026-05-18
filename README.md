# Pixel-level-Training-Of-Image-Segmentation
计算机视觉 HW2 Task3

# 配置环境
```bash
pip install -r requirements.txt
```
# 模型下载
https://drive.google.com/drive/folders/1RUvwHcV9-cDFWHC3W6JsZqqreM_GuV3G?usp=drive_link

# 1.训练测试
```bash
python train.py --epochs 50 --batch_size 16 --image_size 256 --lr 0.0005 --save_dir checkpoints/testrun
```
训练参数均可更改

# 2.评估测试
首先更改evaluate.py内main函数中loaders的batchsize和imgsize大小，以对应将要测试的模型参数，其中ex模型对应16，256*256；fullrun对应64，128*128.
然后更改experiments参数，将其更改为所下载的参数的地址，随后运行
```bash
python evaluate.py
```
# 3.混淆矩阵和错例


更改confusion_matrix.py，main中的WEIGHT_PATH等参数，随后运行
```bash
python confusion_matrix.py
```
