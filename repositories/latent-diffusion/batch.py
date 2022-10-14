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

pathInput = os.getcwd()


def check_model_exists():
    # check if model and yaml exist
    path = pathInput + "/models/ldm/ld_sr".replace('\\', os.sep).replace('/', os.sep)
    model = 'model.ckpt'
    yaml = 'project.yaml'
    if os.path.exists(path):
        if os.path.exists(os.path.join(path, yaml)):
            print('YAML found')
            if os.path.exists(os.path.join(path, model)):
                print('Model found')
                return os.path.join(path, model), os.path.join(path, yaml)
            else:
                return False


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return {"model": model}, global_step


def get_model(mode):
    check = check_model_exists()
    if check != False:
        path_ckpt = check[0]
        path_conf = check[1]
    else:
        print('Model not found, please run the bat file to download the model')
    config = OmegaConf.load(path_conf)
    model, step = load_model_from_config(config, path_ckpt)
    return model


def get_cond_options(mode):
    path = "data/example_conditioning"
    path = os.path.join(path, mode)
    onlyfiles = [f for f in sorted(os.listdir(path))]
    return path, onlyfiles


def get_cond(mode, selected_path):
    example = dict()
    if mode == "superresolution":
        up_f = 4
        c = Image.open(selected_path).convert('RGB')
        c = torch.unsqueeze(torchvision.transforms.ToTensor()(c), 0)
        c_up = torchvision.transforms.functional.resize(c, size=[up_f * c.shape[2], up_f * c.shape[3]], antialias=True)
        c_up = rearrange(c_up, '1 c h w -> 1 h w c')
        c = rearrange(c, '1 c h w -> 1 h w c')
        c = 2. * c - 1.

        c = c.to(torch.device("cuda"))
        example["LR_image"] = c
        example["image"] = c_up

    return example


def run(model, selected_path, task, custom_steps, eta, resize_enabled=False, classifier_ckpt=None, global_step=None):
    # global stride

    example = get_cond(task, selected_path)

    save_intermediate_vid = False
    n_runs = 1
    masked = False
    guider = None
    ckwargs = None
    mode = 'ddim'
    ddim_use_x0_pred = False
    temperature = 1.
    eta = eta
    make_progrow = True
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

    invert_mask = False

    x_T = None
    for n in range(n_runs):
        if custom_shape is not None:
            x_T = torch.randn(1, custom_shape[1], custom_shape[2], custom_shape[3]).to(model.device)
            x_T = repeat(x_T, '1 c h w -> b c h w', b=custom_shape[0])

        logs = make_convolutional_sample(example, model,
                                         mode=mode, custom_steps=custom_steps,
                                         eta=eta, swap_mode=False, masked=masked,
                                         invert_mask=invert_mask, quantize_x0=False,
                                         custom_schedule=None, decode_interval=10,
                                         resize_enabled=resize_enabled, custom_shape=custom_shape,
                                         temperature=temperature, noise_dropout=0.,
                                         corrector=guider, corrector_kwargs=ckwargs, x_T=x_T,
                                         save_intermediate_vid=save_intermediate_vid,
                                         make_progrow=make_progrow, ddim_use_x0_pred=ddim_use_x0_pred
                                         )
    return logs


@torch.no_grad()
def convsample_ddim(model, cond, steps, shape, eta=1.0, callback=None, normals_sequence=None,
                    mask=None, x0=None, quantize_x0=False, img_callback=None,
                    temperature=1., noise_dropout=0., score_corrector=None,
                    corrector_kwargs=None, x_T=None, log_every_t=None
                    ):
    ddim = DDIMSampler(model)
    bs = shape[0]  # dont know where this comes from but wayne
    shape = shape[1:]  # cut batch dim
    print(f"Sampling with eta = {eta}; steps: {steps}")
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, conditioning=cond, callback=callback,
                                         normals_sequence=normals_sequence, quantize_x0=quantize_x0, eta=eta,
                                         mask=mask, x0=x0, temperature=temperature, verbose=False,
                                         score_corrector=score_corrector,
                                         corrector_kwargs=corrector_kwargs, x_T=x_T)

    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(batch, model, mode="vanilla", custom_steps=None, eta=1.0, swap_mode=False, masked=False,
                              invert_mask=True, quantize_x0=False, custom_schedule=None, decode_interval=1000,
                              resize_enabled=False, custom_shape=None, temperature=1., noise_dropout=0., corrector=None,
                              corrector_kwargs=None, x_T=None, save_intermediate_vid=False, make_progrow=True,
                              ddim_use_x0_pred=False):
    log = dict()

    z, c, x, xrec, xc = model.get_input(batch, model.first_stage_key,
                                        return_first_stage_outputs=True,
                                        force_c_encode=not (hasattr(model, 'split_input_params')
                                                            and model.cond_stage_key == 'coordinates_bbox'),
                                        return_original_cond=True)

    log_every_t = 1 if save_intermediate_vid else None

    if custom_shape is not None:
        z = torch.randn(custom_shape)
        # print(f"Generating {custom_shape[0]} samples of shape {custom_shape[1:]}")

    z0 = None

    log["input"] = x
    log["reconstruction"] = xrec

    if ismap(xc):
        log["original_conditioning"] = model.to_rgb(xc)
        if hasattr(model, 'cond_stage_key'):
            log[model.cond_stage_key] = model.to_rgb(xc)

    else:
        log["original_conditioning"] = xc if xc is not None else torch.zeros_like(x)
        if model.cond_stage_model:
            log[model.cond_stage_key] = xc if xc is not None else torch.zeros_like(x)
            if model.cond_stage_key == 'class_label':
                log[model.cond_stage_key] = xc[model.cond_stage_key]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        img_cb = None

        sample, intermediates = convsample_ddim(model, c, steps=custom_steps, shape=z.shape,
                                                eta=eta,
                                                quantize_x0=quantize_x0, img_callback=img_cb, mask=None, x0=z0,
                                                temperature=temperature, noise_dropout=noise_dropout,
                                                score_corrector=corrector, corrector_kwargs=corrector_kwargs,
                                                x_T=x_T, log_every_t=log_every_t)
        t1 = time.time()

        if ddim_use_x0_pred:
            sample = intermediates['pred_x0'][-1]

    x_sample = model.decode_first_stage(sample)

    try:
        x_sample_noquant = model.decode_first_stage(sample, force_not_quantize=True)
        log["sample_noquant"] = x_sample_noquant
        log["sample_diff"] = torch.abs(x_sample_noquant - x_sample)
    except:
        pass

    log["sample"] = x_sample
    log["time"] = t1 - t0

    return log


diffMode = 'superresolution'
model = get_model('superresolution')
import_method = 'Directory'
output_subfolder_name = 'processed'

save_output_to_drive = False
zip_if_not_drive = False

os.makedirs(pathInput + '/content/input'.replace('\\', os.sep).replace('/', os.sep), exist_ok=True)
output_directory = os.getcwd() + f'/content/output/{output_subfolder_name}'.replace('\\', os.sep).replace('/', os.sep)
os.makedirs(output_directory, exist_ok=True)
uploaded_img = pathInput + '/content/input/'.replace('\\', os.sep).replace('/', os.sep)
pathInput, dirsInput, filesInput = next(os.walk(pathInput + '/content/input'))
file_count = len(filesInput)
print(f'Found {file_count} files total')

# Run settings
diffusion_steps = "100"  # @param [25, 50, 100, 250, 500, 1000]
diffusion_steps = int(diffusion_steps)
eta = 1.0  # @param  {type: 'raw'}
stride = 0  # not working atm

# ####Scaling options:
# Down-sampling to 256px first will often improve the final image and runs faster.

# You can improve sharpness without up-scaling by up-scaling and then down-sampling to the original size (i.e. Super 
# Resolution) 
pre_down_sample = 'None'  # @param ['None', '1/2', '1/4']

post_down_sample = 'None'  # @param ['None', 'Original Size', '1/2', '1/4']

# Nearest gives sharper results, but may look more pixellated. Lancoz is much higher quality, but result may be less 
# crisp. 
down_sample_method = 'Lanczos'  # @param ['Nearest', 'Lanczos']

overwrite_prior_runs = True  # @param {type: 'boolean'}

pathProcessed, dirsProcessed, filesProcessed = next(os.walk(output_directory))

for img in filesInput:
    if img in filesProcessed and overwrite_prior_runs is False:
        print(f'Skipping {img}: Already processed')
        continue
    gc.collect()
    torch.cuda.empty_cache()
    dir = pathInput
    filepath = os.path.join(dir, img).replace('\\', os.sep).replace('/', os.sep)

    im_og = Image.open(filepath)
    width_og, height_og = im_og.size

    # Downs-ample Pre
    if pre_down_sample == '1/2':
        down_sample_rate = 2
    elif pre_down_sample == '1/4':
        down_sample_rate = 4
    else:
        down_sample_rate = 1

    width_down_sampled_pre = width_og // down_sample_rate
    height_down_sampled_pre = height_og // down_sample_rate
    if down_sample_rate != 1:
        print(f'Down-sampling from [{width_og}, {height_og}] to [{width_down_sampled_pre}, {height_down_sampled_pre}]')
        im_og = im_og.resize((width_down_sampled_pre, height_down_sampled_pre), Image.LANCZOS)
        im_og.save(dir + '/content/temp.png'.replace('\\', os.sep).replace('/', os.sep))
        filepath = dir + '/content/temp.png'.replace('\\', os.sep).replace('/', os.sep)

    logs = run(model["model"], filepath, diffMode, diffusion_steps, eta)

    sample = logs["sample"]
    sample = sample.detach().cpu()
    sample = torch.clamp(sample, -1., 1.)
    sample = (sample + 1.) / 2. * 255
    sample = sample.numpy().astype(np.uint8)
    sample = np.transpose(sample, (0, 2, 3, 1))
    print(sample.shape)
    a = Image.fromarray(sample[0])

    # Downsample Post
    if post_down_sample == '1/2':
        down_sample_rate = 2
    elif post_down_sample == '1/4':
        down_sample_rate = 4
    else:
        down_sample_rate = 1

    width, height = a.size
    width_down_sampled_post = width // down_sample_rate
    height_down_sampled_post = height // down_sample_rate

    if down_sample_method == 'Lanczos':
        aliasing = Image.Resampling.LANCZOS
    else:
        aliasing = Image.Resampling.NEAREST

    if down_sample_rate != 1:
        print(f'Down-sampling from [{width}, {height}] to [{width_down_sampled_post}, {height_down_sampled_post}]')
        a = a.resize((width_down_sampled_post, height_down_sampled_post), aliasing)
    elif post_down_sample == 'Original Size':
        print(f'Down-sampling from [{width}, {height}] to Original Size [{width_og}, {height_og}]')
        a = a.resize((width_og, height_og), aliasing)

    a.save(f'{output_directory}/{img}')
    gc.collect()
    torch.cuda.empty_cache()

print(f'Processing finished!')
