# Fork for use with [sd-webui](https://github.com/sd-webui/stable-diffusion-webui/)
- If you're following the steps outline via [Installation Link](https://github.com/sd-webui/stable-diffusion-webui/wiki/Installation), there is an optional steps to load Latent Diffusion Super Resolution (LDSR)
- This repo helps minor updates to automatic download_mode.bat and keep things compatible with [sd-webui](https://github.com/sd-webui/stable-diffusion-webui/) 
- download_mode.bat - will download the required model files and place them under experiments/pretrained_models

### LDSR
1. Git clone this repo [Hafiidz/latent-diffusion](https://github.com/Hafiidz/latent-diffusion) into your `/stable-diffusion-webui/src/` folder
2. You can do this by navigating to `/stable-diffusion-webui/src/` folder in your VSCode terminal, or via running [git bash](https://user-images.githubusercontent.com/3688500/189250949-2d07dd66-1612-453f-ae23-5f7cd212f72d.png) or other relevant command prompt

<div align="center">
  <img src=https://user-images.githubusercontent.com/3688500/189254107-2fbae6dc-e856-4814-89e3-256c2b890f30.png  />
</div>

3. Run `git clone https://github.com/Hafiidz/latent-diffusion.git`
4. It will take a while to download and once the cloning is completed, you will see the following messages:
<div align="center">
  <img src=https://user-images.githubusercontent.com/3688500/189251749-27b0563c-1e71-43f9-985c-858f27920fd1.png  />
</div>

5. Next, run `/stable-diffusion-webui/src/latent-diffusion/download_model.bat`, by double clicking it, to automatically download 2 relevant files `project.yaml` and `model.cpkt`
6. The file will take a while to download, once done, navigate to `stable-diffusion-webui/src/latent-diffusion/experiments/pretrained_models/` and confirm the two files are there

<div align="center">
  <img src=https://user-images.githubusercontent.com/3688500/189252740-1aee29fb-7f2a-4873-90b7-74410a5277e8.png   />
</div>

7. _(Optional)_ If the two files are not there, you can manually download them:
    1. Download [project.yaml](https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1) and [model last.cpkt](https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1). 
    2. Rename `<date>-project.yaml` to `project.yaml` and `last.ckpt` to `model.ckpt`
    3. Place both under `stable-diffusion-webui/src/latent-diffusion/experiments/pretrained_models/`
8. Follow the discussion or raise a new issue [here](https://github.com/sd-webui/stable-diffusion-webui/issues/488). 


# Link to Latent diffusion details
- [Original Readme](README_LD.md)
- [CompVis](https://github.com/CompVis/latent-diffusion)

