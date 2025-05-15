# LivePortrait for ComfyUI

安装准备：
1、下面模型（百度云链接: https://pan.baidu.com/s/1-OKeUq57JDiLQj0fxRY9NA 提取码: ncmr，或者到源项目下载），放到comfyui/models/liveportrait，目录结构如下：
```
├── liveportrait
    ├── animals
    │   ├── base_models
    │   │   ├── spade_generator.pth
    │   │   ├── warping_module.pth
    │   │   ├── appearance_feature_extractor.pth
    │   │   ├── motion_extractor.pth
    │   ├── xpose.pth
    ├── landmark.onnx
    ├── insightface
    │   ├── models
    │   │   ├── buffalo_l
    │   │   │   ├── 2d106det.onnx
    │   │   │   ├── det_10g.onnx
    │   │   │   ├── genderage.onnx
    │   │   │   ├── w600k_r50.onnx
    │   │   │   ├── 1k3d68.onnx
    │   │   ├── buffalo_l.zip
    ├── base_models
    │   ├── spade_generator.pth
    │   ├── warping_module.pth
    │   ├── appearance_feature_extractor.pth
    │   ├── motion_extractor.pth

```

2、执行如下命令安装相关依赖包：
```
pip install -r requirements.txt

# 
```
注：如果需要驱动动物，则需要到ComfyUI-liveportrait-fg/liveportrait/src/utils/dependencies/XPose/models/UniPose/ops目录下执行编译命令：
```
# gcc<=13、g++<=13
python setup.py build install
```
如果是linux系统，cuda12.6+python 3.11的，可以通过目录下的multiscaledeformableattention-1.0-cp311-cp311-linux_x86_64.whl安装，本人测试环境使用的。

3、效果如下（工作流在workflows目录下）：
![02](https://github.com/user-attachments/assets/47f25f36-c29d-468d-ad98-93bb31ce03ef)

https://github.com/user-attachments/assets/c6acfb64-9e5d-4ac2-940d-5697d51400d4

https://github.com/user-attachments/assets/772ef3dc-4bd5-4066-8ead-b32516743b28

源项目地址：https://github.com/KwaiVGI/LivePortrait
