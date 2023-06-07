import base64
import io
import time
import datetime
import uvicorn
import math
import gradio as gr
from threading import Lock
from io import BytesIO
from fastapi import APIRouter, Depends, FastAPI, Request, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from secrets import compare_digest

import modules.shared as shared
from modules import sd_samplers, deepbooru, sd_hijack, images, scripts, ui, postprocessing, sd_samplers_common
from modules.api.models import *
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images
from modules.textual_inversion.textual_inversion import create_embedding, train_embedding
from modules.textual_inversion.preprocess import preprocess
from modules.hypernetworks.hypernetwork import create_hypernetwork, train_hypernetwork
from PIL import PngImagePlugin,Image
from modules.sd_models import checkpoints_list, unload_model_weights, reload_model_weights
from modules.sd_models_config import find_checkpoint_config_near_filename
from modules.realesrgan_model import get_realesrgan_models
from modules import devices
from typing import List
import piexif
import piexif.helper
import numpy as np
import torch
import matplotlib.pyplot as plt

import importlib
import hashlib

from modules.shared import opts, state
from modules import sd_samplers, deepbooru, sd_hijack, images, scripts, ui, postprocessing, sd_samplers_common
from modules.processing import create_random_tensors, opt_C, opt_f
from tqdm.auto import trange, tqdm
utils = importlib.import_module("repositories.k-diffusion.k_diffusion.utils")
sampling = importlib.import_module("repositories.k-diffusion.k_diffusion.sampling")

cnet_external_code = importlib.import_module("extensions.sd-webui-controlnet.scripts.external_code")
cnet_global_state = importlib.import_module("extensions.sd-webui-controlnet.scripts.global_state")

cnet_enabled = {
    'canny': {'modules':['canny'], 'model':''},
    'mlsd': {'modules':['mlsd'], 'model':''},
    'depth': {'modules':['depth', 'depth_leres', 'depth_leres++'], 'model':''},
    'normalbae': {'modules':['normal_bae'], 'model':''},
    'seg': {'modules':['segmentation'], 'model':''},
    'inpaint': {'modules':['inpaint'], 'model':''},
    'lineart': {'modules':['lineart', 'lineart_coarse', 'lineart_standard'], 'model':''},
    'openpose': {'modules':['openpose', 'openpose_hand', 'openpose_face', 'openpose_faceonly', 'openpose_full'], 'model':''},
    'scribble': {'modules':['scribble_xdog', 'scribble_hed'], 'model':''},
    'softedge': {'modules':['softedge_hed', 'softedge_hedsafe', 'softedge_pidinet', 'softedge_pidisafe'], 'model':''},
    'shuffle': {'modules':['shuffle'], 'model':''},
    'ip2p': {'modules':[''], 'model':''},
    'tile': {'modules':['tile_resample'], 'model':''},
    'temporal': {'modules':['temporal'], 'model':''},
}

cnet_models = {}

cnet_global_state.update_cn_models()
for m in cnet_global_state.cn_models:
  for cnet_module in cnet_enabled:
    if cnet_enabled[cnet_module] != False and cnet_module in m:
      cnet_enabled[cnet_module]['model'] = m
      for cnet_module_name in cnet_enabled[cnet_module]['modules']:
        cnet_models[cnet_module_name] = m
        print(cnet_module_name, m)

def append_cnet_units(units=[], controlnets=[], **kwargs):

  cnet_args_from = cnet_args_to = len(units)

  for cnet_module in controlnets:

    cnet_images = controlnets[cnet_module]['images']

    cnet_weight = 1.0
    if 'weight' in controlnets[cnet_module]:
      cnet_weight = controlnets[cnet_module]['weight']

    module = cnet_module
    if 'module' in controlnets[cnet_module]:
      module=controlnets[cnet_module]['module']

    if cnet_module == 'temporal':
      module = None

    for cnet_image in cnet_images:
      print("====> append_cnet_units", cnet_module, cnet_models[cnet_module], module)

      units.append(
        cnet_external_code.ControlNetUnit(
          module=module,
          model=cnet_models[cnet_module],
          weight=cnet_weight,
          image=np.array(cnet_image),
        )
      )
      cnet_args_to = cnet_args_to + 1

  return units, cnet_args_from, cnet_args_to

def plot_losses(losses):
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Time')
    plt.show()

def hash_tensor(tensor):
    tensor_np = tensor.detach().cpu().numpy()
    tensor_bytes = tensor_np.tobytes()
    hash_value = hashlib.sha256(tensor_bytes).hexdigest()
    return hash_value

def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
  if predict_cids:
      if z.dim() == 4:
          z = torch.argmax(z.exp(), dim=1).long()
      z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
      z = rearrange(z, 'b h w c -> b c h w').contiguous()

  z = 1. / self.scale_factor * z

  with torch.enable_grad():
    d = self.first_stage_model.decode(z)
    return d

class StableDiffusionProcessingImg2ImgVCN(StableDiffusionProcessingImg2Img):

  def __init__(self,
               vcn_flows = [],
               vcn_previous_frames = [],
               vcn_max_epochs = 150,
               vcn_stop_after_inefficient_steps = 20,
               vcn_optimizer_lr = 0.01,
               vcn_scheduler_factor = 0.1,
               vcn_scheduler_patience = 5,
               **kwargs):

    super().__init__(**kwargs)

    self.vcn_flows = vcn_flows
    self.vcn_max_epochs = vcn_max_epochs
    self.vcn_previous_frames = vcn_previous_frames
    self.vcn_stop_after_inefficient_steps = vcn_stop_after_inefficient_steps
    self.vcn_optimizer_lr = vcn_optimizer_lr
    self.vcn_scheduler_factor = vcn_scheduler_factor
    self.vcn_scheduler_patience = vcn_scheduler_patience

  def init(self, all_prompts, all_seeds, all_subseeds):
    super().init(all_prompts, all_seeds, all_subseeds)

    self.sampler = sd_samplers.create_sampler(self.sampler_name, self.sd_model)
    self.sampler.orig_func = self.sampler.func
    self.sampler.func = torch.enable_grad()(lambda model, x, sigmas, *args, **kwargs: self.sampler.orig_func.__wrapped__(model, x, sigmas, *args, **kwargs))


  def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
      x = create_random_tensors([opt_C, self.height // opt_f, self.width // opt_f], seeds=seeds, subseeds=subseeds, subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)

      if self.vcn_flows != None and len(self.vcn_flows) > 0:
          x = self.temporal_consistency_optimization(x,
                                                      conditioning,
                                                      unconditional_conditioning,
                                                      prompts,
                                                      )

      if self.initial_noise_multiplier != 1.0:
          self.extra_generation_params["Noise multiplier"] = self.initial_noise_multiplier
          x *= self.initial_noise_multiplier

      samples = self.sampler.sample_img2img(self, self.init_latent, x, conditioning, unconditional_conditioning, image_conditioning=self.image_conditioning)

      if self.mask is not None:
          samples = samples * self.nmask + self.init_latent * self.mask

      del x
      devices.torch_gc()

      return samples

  def flow_warping ( self, frame , flow ):
      """
      Warp ( t ) th frame to (t -1) th frame
      args :
      frame : float tensor in [H , W , 3] format
      flow : float tensor in [H , W , 2] format
      """
      # create a matrix where each entry is its coordinate
      s = flow . shape [:2]
      m = np . array ([ i for i in np . ndindex (* s )])
      meshgrid = torch . from_numpy ( m . reshape (* s , len ( s )))

      # find the coordinates that the flows point to
      dest = flow + meshgrid

      # discard out - of - frame flows
      valid_mask = (( dest >= 0) &
      ( dest < torch . tensor ( dest . shape [:2]) - 1)). all ( -1)

      v_src = meshgrid [ valid_mask ]

      # nearest - neighbor warping
      v_dst = dest [ valid_mask ]
      v_dst = v_dst . round (). to ( int )

      valid_pixels = frame [ v_dst [: , 0] , v_dst [: , 1]]
      frame [ v_src [: , 0] , v_src [: , 1]] = valid_pixels

      return frame

  def temporal_consistency_optimization(self, noise, conditioning, unconditional_conditioning, prompts):
    """
    Craft an optimal noise to generate temporally consistent video
    args :
    noise : float tensor in [T , H , W , C ] format
    conditioning : float tensor in [T , H , W , C ] format
    unconditional_conditioning : float tensor in [T , H , W , C ] format
    prompts : textual conditioning strings
    n_epochs : number of epochs of the optimization
    flow : float tensor in [T -1 , H , W , 2] format
    """
    self.loss_history = []

    minimal_loss = None
    optimal_noise = noise

    noise.requires_grad_(True)
    self.init_latent.requires_grad_(True)
    optimizer = torch.optim.AdamW([noise], lr=self.vcn_optimizer_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.vcn_scheduler_factor, patience=self.vcn_scheduler_patience)

    for epoch in range (self.vcn_max_epochs):

      optimizer.zero_grad ()

      with torch.enable_grad():
        samples_ddim = self.sampler.sample_img2img(self,
          self.init_latent,
          noise,
          conditioning,
          unconditional_conditioning,
          image_conditioning=self.image_conditioning)

        x_samples_ddim = [decode_first_stage(self.sd_model, samples_ddim)[0]]
        x_samples_ddim = torch.stack(x_samples_ddim).float()
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

        x_sample = x_samples_ddim[0] * 255.0
        x_sample = x_sample.permute(1, 2, 0)

        ref = self.vcn_previous_frames[0]

        warped = self.flow_warping ( x_sample , self.vcn_flows[0])

        loss = []

        err = torch . where ( warped != 0 , warped - ref , 0) ** 2

        # normalized by number of non - zero pixels
        loss . append ( err . sum () / ( err !=0). sum ())
        loss = sum ( loss )/ len ( loss )
        self.loss_history.append(loss.item())

        if minimal_loss == None or loss < minimal_loss:
          print("\n===> setting minimal loss", loss, minimal_loss)
          minimal_loss = loss
          optimal_noise = noise.clone()

      loss.backward ()
      optimizer.step ()
      scheduler.step(loss)

      if min(self.loss_history[-self.vcn_stop_after_inefficient_steps:]) > minimal_loss:
        break

      current_lr = optimizer.param_groups[0]['lr']
      print("\n===> loss", epoch, loss.item(), current_lr, hash_tensor(noise))

    print("\n====> final", minimal_loss, hash_tensor(optimal_noise))
    return optimal_noise


def infer(controlnets=[],
          vcn_flows = [],
          vcn_previous_frames = [],
          vcn_max_epochs = 150,
          vcn_stop_after_inefficient_steps = 20,
          vcn_optimizer_lr = 0.1,
          vcn_scheduler_factor = 0.1,
          vcn_scheduler_patience = 5,
          **kwargs):

  args = {}
  script_args = {}

  units = []
  units, cnet_args_from, cnet_args_to = append_cnet_units(units, controlnets, **kwargs)

  script_args = tuple(units)

  p = StableDiffusionProcessingImg2ImgVCN(
      sd_model=shared.sd_model,
      do_not_save_samples=True,

      vcn_flows = vcn_flows,
      vcn_max_epochs = vcn_max_epochs,
      vcn_previous_frames = vcn_previous_frames,
      vcn_stop_after_inefficient_steps = vcn_stop_after_inefficient_steps,
      vcn_optimizer_lr = vcn_optimizer_lr,
      vcn_scheduler_factor = vcn_scheduler_factor,
      vcn_scheduler_patience = vcn_scheduler_patience,

      **kwargs
  )

  p.script_args = tuple(script_args)
  p.scripts = scripts.scripts_txt2img
  p.scripts.initialize_scripts(False)

  enabled_scripts = []

  for index in range(len(p.scripts.scripts)):
    script = p.scripts.scripts[index]

    if script.title() == 'ControlNet':
      p.scripts.scripts[index].args_from = cnet_args_from
      p.scripts.scripts[index].args_to = cnet_args_to

      enabled_scripts.append(p.scripts.scripts[index])

  p.scripts.scripts = enabled_scripts
  p.scripts.alwayson_scripts = enabled_scripts

  shared.state.begin()
  processed = process_images(p)
  shared.state.end()

  p.close()
  shared.total_tqdm.clear()

  return processed.images


def engrid(images):
  power = int(math.sqrt(len(images)))
  img = images[0]
  dimx = img.width
  dimy = img.height

  grid = Image.new(size=(dimx*power, dimy*power), mode='RGB')

  x = 0
  y = 0
  for img in images:
    grid.paste(img, (x*dimx, y*dimy))
    x = x + 1
    if x == power:
      x = 0
      y = y + 1

  return grid

def degrid(grid, power):
  dimx = int(grid.width / power)
  dimy = int(grid.height / power)
  images = []
  for y in range(power):
    for x in range(power):
      img = grid.crop((x*dimx, y*dimy, (x+1)*dimx, (y+1)*dimy))
      images.append(img)
  return images
