# DreamArtist
This repo is the official PyTorch implementation of ***"DreamArtist: Towards Controllable One-Shot Text-to-Image Generation via Contrastive Prompt-Tuning"*** 
with [Stable-Diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

***Stable-Diffusion-webui Extension Version :*** [DreamArtist-sd-webui-extension](https://github.com/7eu7d7/DreamArtist-sd-webui-extension)

Everyone is an artist. Rome wasn't built in a day, but your artist dreams can be!

With just ***one*** training image DreamArtist learns the content and style in it, generating diverse high-quality images with high controllability.
Embeddings of DreamArtist can be easily combined with additional descriptions, as well as two learned embeddings.

![](imgs/exp1.jpg)
![](imgs/exp_text1.jpg)
![](imgs/exp_text2.jpg)
![](imgs/exp_text3.jpg)

# Setup and Running
Clone this repo.
```bash
git clone https://github.com/7eu7d7/DreamArtist-stable-diffusion
```

Following the [instructions of webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui#automatic-installation-on-windows) to install.

## Training and Usage

First create the positive and negative embeddings in ```DreamArtist Create Embedding``` Tab.
![](imgs/create.jpg)

### Preview Setting
After that, the ```names``` of the positive and negative embedding (```{name}``` and ```{name}-neg```) should be filled into the
```txt2img Tab``` with some common descriptions. This will ensure a correct preview image.
![](https://github.com/7eu7d7/DreamArtist-sd-webui-extension/blob/master/imgs/preview.png)

### Train
Then, select positive embedding and set the parameters and image folder path in the ```Train``` Tab to start training.
The corresponding negative embedding is loaded automatically.
If your VRAM is low or you want save time, you can uncheck the ```reconstruction```.

[Recommended parameters](https://github.com/7eu7d7/DreamArtist-sd-webui-extension#pre-trained-embeddings)

***better to train without filewords***
![](imgs/train.jpg)

Remember to check the option below, otherwise the preview is wrong.
![](https://github.com/7eu7d7/DreamArtist-sd-webui-extension/blob/master/imgs/fromtxt.png)

### Inference
Fill the trained positive and negative embedding into txt2img to generate with DreamArtist prompt.
![](https://github.com/7eu7d7/DreamArtist-sd-webui-extension/blob/master/imgs/gen.jpg)

## Tested models (need ema version):
+ Stable Diffusion v1.5
+ animefull-latest
+ Anything v3.0

Embeddings can be transferred between different models of the same dataset.

## Pre-trained embeddings:

[Download](https://github.com/7eu7d7/DreamArtist-stable-diffusion/releases/tag/embeddings_v2)

| Name       | Model            | Image                                                              | embedding length <br> (Positive, Negative) | iter  | lr     | cfg scale |
|------------|------------------|--------------------------------------------------------------------|--------------------------------------------|-------|--------|-----------|
| ani-nahida | animefull-latest | <img src="https://github.com/7eu7d7/DreamArtist-sd-webui-extension/blob/master/imgs/pre/nahida.jpg" width = "80" height = "80" alt=""/> | 3, 6                                       | 8000  | 0.0025 | 3         |
| ani-cocomi | animefull-latest | <img src="https://github.com/7eu7d7/DreamArtist-sd-webui-extension/blob/master/imgs/pre/cocomi.jpg" width = "80" height = "80" alt=""/> | 3, 6                                       | 8000  | 0.0025 | 3         |
| ani-gura   | animefull-latest | <img src="https://github.com/7eu7d7/DreamArtist-sd-webui-extension/blob/master/imgs/pre/gura.jpg" width = "80" height = "80" alt=""/>   | 3, 6                                       | 12000 | 0.0025 | 3         |
| ani-g      | animefull-latest | <img src="https://github.com/7eu7d7/DreamArtist-sd-webui-extension/blob/master/imgs/pre/g.jpg" width = "80" height = "80" alt=""/>      | 3, 10                                      | 1500  | 0.003  | 5         |
| asty-bk    | animefull-latest | <img src="https://github.com/7eu7d7/DreamArtist-sd-webui-extension/blob/master/imgs/pre/bk.jpg" width = "80" height = "80" alt=""/>     | 3, 6                                       | 5000  | 0.003  | 3         |
| asty-gc    | animefull-latest | <img src="https://github.com/7eu7d7/DreamArtist-sd-webui-extension/blob/master/imgs/pre/gc.jpg" width = "80" height = "80" alt=""/>     | 3, 10                                      | 1000  | 0.005  | 5         |
| real-dog   | sd v1.4          | <img src="https://github.com/7eu7d7/DreamArtist-sd-webui-extension/blob/master/imgs/pre/dog.jpg" width = "80" height = "80" alt=""/>    | 3, 3                                       | 1000  | 0.005  | 5         |
| real-sship | sd v1.4          | <img src="https://github.com/7eu7d7/DreamArtist-sd-webui-extension/blob/master/imgs/pre/sship.jpg" width = "80" height = "80" alt=""/>  | 3, 3                                       | 3000  | 0.003  | 5         |
| sty-cyber  | sd v1.4          | <img src="https://github.com/7eu7d7/DreamArtist-sd-webui-extension/blob/master/imgs/pre/cyber.jpg" width = "80" height = "80" alt=""/>  | 3, 5                                       | 15000 | 0.0025 | 5         |
| sty-shuimo | sd v1.4          | <img src="https://github.com/7eu7d7/DreamArtist-sd-webui-extension/blob/master/imgs/pre/shuimo.jpg" width = "80" height = "80" alt=""/> | 3, 5                                       | 15000 | 0.0025 | 5         |


# Style Clone
![](imgs/exp_style.jpg)

# Prompt Compositions
![](imgs/exp_comp.jpg)

# Comparison on One-Shot Learning
![](imgs/cmp.jpg)

# Other Results
![](imgs/cnx.jpg)
![](imgs/cnx2.jpg)