# 高密度环境下的行人统计

本项目目标检测算法使用的是[**FCHD-Fully-Convolutional-Head-Detector**](https://github.com/aditya-vora/FCHD-Fully-Convolutional-Head-Detector)，目标跟踪使用的是在线多实例学习online MIL算法。



## 依赖
- 系统Windows 10，显卡使用的GTX 1060，理论上Linux也可以运行。

- 安装PyTorch >=0.4 with GPU (code are GPU-only)

- 安装 cupy,可以通过 `pip install cupy-cuda80` or(cupy-cuda90,cupy-cuda91, etc).

- install visdom for visualization, refer to their [github page](https://github.com/facebookresearch/visdom)

## 

## 运行

1. 下载 VGG16 预训练模型，[link](https://drive.google.com/open?id=10AwNitG-5gq-YEJcG9iihosiOu7vAnfO).下载好后存到`data/pretrained_model ` folder.
2. 下载训练好的模型 [link](https://drive.google.com/open?id=1DbE4tAkaFYOEItwuIQhlbZypuIPDrArM). 放到`checkpoints/ ` folder. 
3. 下载测试视频[video1](https://drive.google.com/open?id=1pTFrt4m2rhKYH68piKI1mlFz9N0Gkk10)，[video2](https://drive.google.com/open?id=1ddOThlTD7jlpJ7IyMm4zIcWNhwrZyJpi)
4. 修改`demo.py`中的视频路径
5. 运行

```shell
python demo.py
```



## 参考

[FCHD-Fully-Convolutional-Head-Detector](https://github.com/aditya-vora/FCHD-Fully-Convolutional-Head-Detector)