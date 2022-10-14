import gc
import os
import time
import warnings

import numpy as np
import torch
import torchvision
from PIL import Image
from einops import rearrange, repeat
from omegaconf import OmegaConf

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from ldm.util import ismap

warnings.filterwarnings("ignore", category=UserWarning)


# Create LDSR Class
class LDSR:

    # init function
    def __init__(self, modelPath, yamlPath):
        self.modelPath = modelPath
        self.yamlPath = yamlPath

    def load_model_from_config(self):
        pl_sd = torch.load(self.modelPath, map_location="cpu")
        sd = pl_sd["state_dict"]
        config = OmegaConf.load(self.yamlPath)
        model = instantiate_from_config(config.model)
        _, _ = model.load_state_dict(sd, strict=False)
        model.cuda()
        model.eval()
        return {"model": model}  # , global_step

    def get_cond_options(self, mode):
        path = "data/example_conditioning"
        path = os.path.join(path, mode)
        only_files = [f for f in sorted(os.listdir(path))]
        return path, only_files

    def run(self, model, selected_path, task, custom_steps, eta):
        def make_convolutional_sample(batch, s_model, s_custom_steps=None, s_eta=1.0, quantize_x0=False,
                                      s_custom_shape=None, s_temperature=1., corrector=None,
                                      corrector_kwargs=None, s_x_t=None, s_ddim_use_x0_pred=False):
            log = dict()

            z, c, x, xrec, xc = s_model.get_input(batch, s_model.first_stage_key,
                                                  return_first_stage_outputs=True,
                                                  force_c_encode=not (hasattr(s_model, 'split_input_params')
                                                                      and s_model.cond_stage_key == 'coordinates_bbox'),
                                                  return_original_cond=True)

            if s_custom_shape is not None:
                z = torch.randn(s_custom_shape)

            z0 = None

            log["input"] = x
            log["reconstruction"] = xrec

            if ismap(xc):
                log["original_conditioning"] = s_model.to_rgb(xc)
                if hasattr(s_model, 'cond_stage_key'):
                    log[s_model.cond_stage_key] = s_model.to_rgb(xc)

            else:
                log["original_conditioning"] = xc if xc is not None else torch.zeros_like(x)
                if s_model.cond_stage_model:
                    log[s_model.cond_stage_key] = xc if xc is not None else torch.zeros_like(x)
                    if s_model.cond_stage_key == 'class_label':
                        log[s_model.cond_stage_key] = xc[s_model.cond_stage_key]

            with s_model.ema_scope("Plotting"):
                t0 = time.time()

                sample, intermediates = conv_sample_ddim(s_model, c, steps=s_custom_steps, shape=z.shape,
                                                         eta=s_eta,
                                                         quantize_x0=quantize_x0, mask=None, x0=z0,
                                                         conv_temperature=s_temperature, score_corrector=corrector, corrector_kwargs=corrector_kwargs,
                                                         conv_x_t=s_x_t)
                t1 = time.time()

                if s_ddim_use_x0_pred:
                    sample = intermediates['pred_x0'][-1]

            x_sample = s_model.decode_first_stage(sample)

            try:
                x_sample_no_quantize = s_model.decode_first_stage(sample, force_not_quantize=True)
                log["sample_no_quantize"] = x_sample_no_quantize
                log["sample_diff"] = torch.abs(x_sample_no_quantize - x_sample)
            except Exception:
                pass

            log["sample"] = x_sample
            log["time"] = t1 - t0

            return log

        def conv_sample_ddim(model, cond, steps, shape, eta=1.0, callback=None, normals_sequence=None,
                             mask=None, x0=None, quantize_x0=False, conv_temperature=1., score_corrector=None,
                             corrector_kwargs=None, conv_x_t=None
                             ):

            ddim = DDIMSampler(model)
            bs = shape[0]  # dont know where this comes from but wayne
            shape = shape[1:]  # cut batch dim
            print(f"Sampling with eta = {eta}; steps: {steps}")
            samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, conditioning=cond,
                                                 callback=callback,
                                                 normals_sequence=normals_sequence, quantize_x0=quantize_x0, eta=eta,
                                                 mask=mask, x0=x0, temperature=conv_temperature, verbose=False,
                                                 score_corrector=score_corrector,
                                                 corrector_kwargs=corrector_kwargs, x_T=conv_x_t)

            return samples, intermediates

        # global stride
        def get_cond(cond_mode, cond_selected_path):
            cond_example = dict()
            if cond_mode == "superresolution":
                up_f = 4
                c = cond_selected_path.convert('RGB')
                c = torch.unsqueeze(torchvision.transforms.ToTensor()(c), 0)
                c_up = torchvision.transforms.functional.resize(c, size=[up_f * c.shape[2], up_f * c.shape[3]],
                                                                antialias=True)
                c_up = rearrange(c_up, '1 c h w -> 1 h w c')
                c = rearrange(c, '1 c h w -> 1 h w c')
                c = 2. * c - 1.

                c = c.to(torch.device("cuda"))
                cond_example["LR_image"] = c
                cond_example["image"] = c_up

            return cond_example

        example = get_cond(task, selected_path)

        save_intermediate_vid = False
        n_runs = 1
        guider = None
        ckwargs = None
        ddim_use_x0_pred = False
        temperature = 1.
        eta = eta
        custom_shape = None

        height, width = example["image"].shape[1:3]
        split_input = height >= 128 and width >= 128

        if split_input:
            ks = 128
            stride = 64
            vqf = 4  #
            model.split_input_params = {"ks": (ks, ks), "stride": (stride, stride),
                                        "vqf": vqf,
                                        "patch_distributed_vq": True,
                                        "tie_braker": False,
                                        "clip_max_weight": 0.5,
                                        "clip_min_weight": 0.01,
                                        "clip_max_tie_weight": 0.5,
                                        "clip_min_tie_weight": 0.01}
        else:
            if hasattr(model, "split_input_params"):
                delattr(model, "split_input_params")

        x_t = None
        logs = None
        for n in range(n_runs):
            if custom_shape is not None:
                x_t = torch.randn(1, custom_shape[1], custom_shape[2], custom_shape[3]).to(model.device)
                x_t = repeat(x_t, '1 c h w -> b c h w', b=custom_shape[0])

            logs = make_convolutional_sample(example, model,
                                             s_custom_steps=custom_steps,
                                             s_eta=eta, quantize_x0=False,
                                             s_custom_shape=custom_shape,
                                             s_temperature=temperature, corrector=guider, corrector_kwargs=ckwargs, s_x_t=x_t,
                                             s_ddim_use_x0_pred=ddim_use_x0_pred
                                             )
        return logs

    @torch.no_grad()
    def super_resolution(self, image, steps=100, pre_down_scale='None', post_down_scale='None'):
        pre_scale = 1
        if pre_down_scale == '1/4':
            pre_scale = 4
        if pre_down_scale == '1/3':
            pre_scale = 3
        if pre_down_scale == '1/2':
            pre_scale = 2

        post_scale = 1
        if post_down_scale == '1/4':
            post_scale = 4
        if post_down_scale == '1/3':
            post_scale = 3
        if post_down_scale == '1/2':
            post_scale = 2

        return self.superResolution(image, steps, pre_scale, post_scale)

    @torch.no_grad()
    def superResolution(self, image, steps=100, pre_down_scale=1, post_down_scale=1):
        diffMode = 'superresolution'
        model = self.load_model_from_config()

        # Run settings
        diffusion_steps = int(steps)  # @param [25, 50, 100, 250, 500, 1000]
        eta = 1.0  # @param  {type: 'raw'}

        # ####Scaling options:
        # Down sampling to 256px first will often improve the final image and runs faster.

        # You can improve sharpness without upscaling by upscaling and then downsampling to the original size (i.e.
        # Super Resolution)
        pre_down_sample = pre_down_scale  # @param ['None', '1/2', '1/4']

        post_down_sample = post_down_scale  # @param ['None', 'Original Size', '1/2', '1/4']

        # Nearest gives sharper results, but may look more pixellated. Lancoz is much higher quality, but result may
        # be less crisp.
        down_sample_method = 'Lanczos'  # @param ['Nearest', 'Lanczos']

        gc.collect()
        torch.cuda.empty_cache()

        im_og = image
        width_og, height_og = im_og.size

        # Down sample Pre
        if pre_down_sample == '1/2':
            downsample_rate = 2
        elif pre_down_sample == '1/3':
            downsample_rate = 3
        elif pre_down_sample == '1/4':
            downsample_rate = 4
        else:
            downsample_rate = 1

        width_downsampled_pre = width_og // downsample_rate
        height_downsampled_pre = height_og // downsample_rate
        if downsample_rate != 1:
            print(f'Downsampling from [{width_og}, {height_og}] to [{width_downsampled_pre}, {height_downsampled_pre}]')
            im_og = im_og.resize((width_downsampled_pre, height_downsampled_pre), Image.LANCZOS)

        logs = self.run(model["model"], im_og, diffMode, diffusion_steps, eta)

        sample = logs["sample"]
        sample = sample.detach().cpu()
        sample = torch.clamp(sample, -1., 1.)
        sample = (sample + 1.) / 2. * 255
        sample = sample.numpy().astype(np.uint8)
        sample = np.transpose(sample, (0, 2, 3, 1))
        a = Image.fromarray(sample[0])

        # Down sample Post
        if post_down_sample == '1/2':
            downsample_rate = 2
        elif post_down_sample == '1/3':
            downsample_rate = 3
        elif post_down_sample == '1/4':
            downsample_rate = 4
        else:
            downsample_rate = 1

        width, height = a.size
        width_downsampled_post = width // downsample_rate
        height_downsampled_post = height // downsample_rate

        if down_sample_method == 'Lanczos':
            aliasing = Image.LANCZOS
        else:
            aliasing = Image.NEAREST

        if downsample_rate != 1:
            print(f'Down sampling from [{width}, {height}] to [{width_downsampled_post}, {height_downsampled_post}]')
            a = a.resize((width_downsampled_post, height_downsampled_post), aliasing)
        elif post_down_sample == 'Original Size':
            print(f'Down sampling from [{width}, {height}] to Original Size [{width_og}, {height_og}]')
            a = a.resize((width_og, height_og), aliasing)

        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f'Processing finished!')
        return a
