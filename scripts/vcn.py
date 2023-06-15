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
from PIL import PngImagePlugin,Image,ImageDraw
from modules.sd_models import checkpoints_list, unload_model_weights, reload_model_weights
from modules.sd_models_config import find_checkpoint_config_near_filename
from modules.realesrgan_model import get_realesrgan_models
from modules import devices
from typing import List
from einops import rearrange

import piexif
import piexif.helper
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2 as cv

import importlib
import hashlib

from modules.shared import opts, state
from modules import sd_samplers, deepbooru, sd_hijack, images, scripts, ui, postprocessing, sd_samplers_common
from modules.processing import create_random_tensors, opt_C, opt_f
from tqdm.auto import trange, tqdm
utils = importlib.import_module("repositories.k-diffusion.k_diffusion.utils")
sampling = importlib.import_module("repositories.k-diffusion.k_diffusion.sampling")

from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow import Raft_Large_Weights
import torchvision.transforms.functional as F
import torch.nn.functional as Fnn

vram_debug = False

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

def load_cnet_models():
  global cnet_models, cnet_enabled

  cnet_global_state = importlib.import_module("extensions.sd-webui-controlnet.scripts.global_state")

  cnet_global_state.update_cn_models()
  for m in cnet_global_state.cn_models:
    for cnet_module in cnet_enabled:
      if cnet_enabled[cnet_module] != False and cnet_module in m:
        cnet_enabled[cnet_module]['model'] = m
        for cnet_module_name in cnet_enabled[cnet_module]['modules']:
          cnet_models[cnet_module_name] = m

def append_cnet_units(units=[], controlnets=[], **kwargs):

  cnet_external_code = importlib.import_module("extensions.sd-webui-controlnet.scripts.external_code")

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

def append_sag_units(units=[], sag_enabled=False, sag_scale=0.75, sag_mask_threshold=1.0, **kwargs):
    try:
        sag_args_from = sag_args_to = len(units)

        if sag_enabled:
            from extensions.sd_webui_SAG.scripts.SAG import SAGUnit
            units.append(
                SAGUnit(
                    enabled=True,
                    scale=sag_scale,
                    mask_threshold=sag_mask_threshold,
                )
            )
            sag_args_to = sag_args_to + 1

        return units, sag_args_from, sag_args_to
    except:
      return units, 0, 0

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
               vcn_key_frame = None,
               vcn_previous_frames = [],
               vcn_max_epochs = 50,
               vcn_optimizer_lr = 0.01,
               vcn_scheduler_factor = 0.1,
               vcn_scheduler_patience = 5,
               vcn_sample_steps = 10,
               vcn_warp_error_scale = 1,
               vcn_flow_error_scale = 1,
               vcn_lineart_error_scale = 1,
               vcn_error_percentile = 0.9,
               vcn_fidelity_oriented_compensation = False,
               vcn_fidelity_oriented_compensation_mask = 0.1,
               vcn_adain = False,
               vcn_blur = False,
               vcn_blur_kernel = 9,
               vcn_blur_sigma = 1.0,
               **kwargs):

    super().__init__(**kwargs)

    if vcn_key_frame == None:
        vcn_key_frame = vcn_previous_frames[0]
    self.vcn_key_frame = vcn_key_frame

    self.vcn_flows = []
    for flow in vcn_flows:
        self.vcn_flows.append(flow.to('cuda'))

    self.vcn_max_epochs = vcn_max_epochs
    self.vcn_previous_frames = vcn_previous_frames
    self.vcn_optimizer_lr = vcn_optimizer_lr
    self.vcn_scheduler_factor = vcn_scheduler_factor
    self.vcn_scheduler_patience = vcn_scheduler_patience
    self.vcn_sample_steps = vcn_sample_steps

    self.vcn_warp_error_scale = vcn_warp_error_scale
    self.vcn_flow_error_scale = vcn_flow_error_scale
    self.vcn_lineart_error_scale = vcn_lineart_error_scale
    self.vcn_error_percentile = vcn_error_percentile

    self.raft_model = None
    self.lineart_detector = None

    self.vcn_fidelity_oriented_compensation = vcn_fidelity_oriented_compensation
    self.vcn_fidelity_oriented_compensation_mask = vcn_fidelity_oriented_compensation_mask
    self.vcn_adain = vcn_adain
    self.vcn_blur = vcn_blur
    self.vcn_blur_kernel = vcn_blur_kernel
    self.vcn_blur_sigma = vcn_blur_sigma

    self.loss_history = []

  def init(self, all_prompts, all_seeds, all_subseeds):
    super().init(all_prompts, all_seeds, all_subseeds)

    if self.vcn_flows != None and len(self.vcn_flows) > 0:
        self.sampler = sd_samplers.create_sampler(self.sampler_name, self.sd_model)
        self.sampler.orig_func = self.sampler.func
        self.sampler.func = torch.enable_grad()(lambda model, x, sigmas, *args, **kwargs: self.sampler.orig_func.__wrapped__(model, x, sigmas, *args, **kwargs))


  def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
      x = create_random_tensors([opt_C, self.height // opt_f, self.width // opt_f],
                                seeds=seeds,
                                subseeds=subseeds,
                                subseed_strength=self.subseed_strength,
                                seed_resize_from_h=self.seed_resize_from_h,
                                seed_resize_from_w=self.seed_resize_from_w,
                                p=self)

      if self.vcn_flows != None and len(self.vcn_flows) > 0:

          max_epochs = [self.vcn_max_epochs]
          if isinstance(self.vcn_max_epochs, list):
              max_epochs = self.vcn_max_epochs

          vcn_flow_error_scale = [self.vcn_flow_error_scale]
          if isinstance(self.vcn_flow_error_scale, list):
              vcn_flow_error_scale = self.vcn_flow_error_scale

          vcn_warp_error_scale = [self.vcn_warp_error_scale]
          if isinstance(self.vcn_warp_error_scale, list):
              vcn_warp_error_scale = self.vcn_warp_error_scale

          vcn_lineart_error_scale = [self.vcn_lineart_error_scale]
          if isinstance(self.vcn_lineart_error_scale, list):
              vcn_lineart_error_scale = self.vcn_lineart_error_scale

          vcn_optimizer_lr = [self.vcn_optimizer_lr]
          if isinstance(self.vcn_optimizer_lr, list):
              vcn_optimizer_lr = self.vcn_optimizer_lr

          self.loss_history = []

          power = 0
          for epochs in max_epochs:
              self.init_latent, x = self.temporal_consistency_optimization(
                                                         self.init_latent.detach(),
                                                         x.detach(),
                                                         conditioning,
                                                         unconditional_conditioning,
                                                         prompts,
                                                         vcn_max_epochs=epochs,
                                                         vcn_optimizer_lr=self.vcn_optimizer_lr[power],
                                                         vcn_flow_error_scale=vcn_flow_error_scale[power],
                                                         vcn_warp_error_scale=vcn_warp_error_scale[power],
                                                         vcn_lineart_error_scale=vcn_lineart_error_scale[power],
                                                         )

              power = power + 1


      if self.initial_noise_multiplier != 1.0:
          self.extra_generation_params["Noise multiplier"] = self.initial_noise_multiplier
          x *= self.initial_noise_multiplier

      if self.vcn_blur:
          self.init_latent = gaussian_blur_2d(self.init_latent,
                                              kernel_size=self.vcn_blur_kernel,
                                              sigma=self.vcn_blur_sigma)

      samples = self.sampler.sample_img2img(self,
                                            self.init_latent,
                                            x,
                                            conditioning,
                                            unconditional_conditioning,
                                            image_conditioning=self.image_conditioning)

      if self.mask is not None:
          samples = samples * self.nmask + self.init_latent * self.mask

      del x
      devices.torch_gc()

      if self.vcn_fidelity_oriented_compensation:
          samples = fidelity_oriented_zeroshot_encoding(samples, self.vcn_fidelity_oriented_compensation_mask)

      if self.vcn_adain:
          prev = torch.tensor([np.array(self.vcn_key_frame)])
          samples = adaptive_instance_normalization(
                  samples,
                  encode(prev.to('cuda'))
                  )

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
      meshgrid = torch.from_numpy(m . reshape (* s , len ( s ))).to('cuda')

      # find the coordinates that the flows point to
      dest = flow + meshgrid

      # discard out - of - frame flows
      valid_mask = ((dest >= 0) &
        (dest < torch.tensor(dest.shape [:2], device='cuda') - 1)).all(-1)

      v_src = meshgrid [ valid_mask ]

      # nearest - neighbor warping
      v_dst = dest [ valid_mask ]
      v_dst = v_dst . round (). to ( int )

      valid_pixels = frame [ v_dst [: , 0] , v_dst [: , 1]]
      frame [ v_src [: , 0] , v_src [: , 1]] = valid_pixels

      return frame

  def calc_error(self,
                 x_sample,
                 ref,
                 ref_flow,
                 ref_lineart,
                 vcn_warp_error_scale,
                 vcn_flow_error_scale,
                 vcn_lineart_error_scale):

        flow = None
        if vcn_flow_error_scale > 0:
            flow = self.get_flow(x_sample, ref)

        warped = None
        if vcn_warp_error_scale > 0:
            warped = self.flow_warping (x_sample, ref_flow)

        lineart = None
        if vcn_lineart_error_scale > 0:
            if warped == None:
                warped = self.flow_warping (x_sample, ref_flow)
            lineart = self.get_lineart(warped)

        err1 = torch.tensor(0).to('cuda')
        if warped != None:
            err1 = (torch . where ( warped != 0 , warped - ref , 0) ** 2).reshape(-1)
            err1 = torch.kthvalue(err1, int(self.vcn_error_percentile * err1.numel())).values # percentile
            print("\n===> warp_err", err1, vcn_warp_error_scale, err1*vcn_warp_error_scale)

        err2 = torch.tensor(0).to('cuda')
        if flow != None:
            err2 = (torch . where ( flow != 0 , ref_flow - flow , 0) ** 2).reshape(-1)
            err2 = torch.kthvalue(err2, int(self.vcn_error_percentile * err2.numel())).values # 90%%
            print("\n===> flow_err", err2, vcn_flow_error_scale, err2*vcn_flow_error_scale)

        err3 = torch.tensor(0).to('cuda')
        if lineart != None:
            err3 = (torch . where ( lineart != 0 , ref_lineart - lineart , 0) ** 2).reshape(-1)
            err3 = torch.kthvalue(err3, int(self.vcn_error_percentile * err3.numel())).values # 90%%
            print("\n===> lineart_err", err3, vcn_lineart_error_scale, err3*vcn_lineart_error_scale)

        err = vcn_warp_error_scale * err1 + vcn_flow_error_scale * err2 + vcn_lineart_error_scale * err3
        print("\n===> err", err)

        return err

  def get_lineart(self, sample):
      print("\n====>vram get_lineart start", torch.cuda.memory_allocated('cuda') / 1024**3) if vram_debug else None
      if self.lineart_detector == None:
         lineart = importlib.import_module("extensions.sd-webui-controlnet.annotator.lineart")

         class LineartDetector(lineart.LineartDetector):
             def __call__(self, input_image):
                if self.model is None:
                    self.load_model(self.model_name)
                self.model.to(self.device)

                assert input_image.ndim == 3
                image = input_image.to('cuda')

                image = image.float()
                image = image / 255.0
                image = rearrange(image, 'h w c -> 1 c h w')
                line = self.model(image)[0][0]
                line = (line * 255.0).clip(0, 255)

                return line

         self.lineart_detector = LineartDetector("sk_model.pth")
         self.lineart_detector.device = 'cuda'
      print("\n====>vram get_lineart end", torch.cuda.memory_allocated('cuda') / 1024**3) if vram_debug else None

      with torch.no_grad():
        return self.lineart_detector(sample)

  def get_flow(self, frame1, frame2):
      print("\n====>vram get_flow start", torch.cuda.memory_allocated('cuda') / 1024**3) if vram_debug else None

      if self.raft_model == None:
          self.raft_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to('cuda')
          self.raft_model = self.raft_model.eval()

      # If you can, run this example on a GPU, it will be a lot faster.
      device = "cuda"

      f1 = frame1.permute(2, 0, 1)/255
      f2 = frame2.permute(2, 0, 1)/255

      f1 = torch.stack([f1])
      f2 = torch.stack([f2])

      f1 = F.resize(f1, size=[frame1.shape[0], frame1.shape[1]], antialias=False)
      f2 = F.resize(f2, size=[frame2.shape[0], frame2.shape[1]], antialias=False)

      weights = Raft_Large_Weights.DEFAULT
      transforms = weights.transforms()

      f1, f2 = transforms(f1, f2)

      list_of_flows = self.raft_model(f2.to(device), f1.to(device))

      flow = list_of_flows[-1].squeeze(0).permute(1,2,0)

      print("\n====>vram get_flow end", torch.cuda.memory_allocated('cuda') / 1024**3) if vram_debug else None

      return flow

  def temporal_consistency_optimization(self,
                                        init_latent,
                                        noise,
                                        conditioning,
                                        unconditional_conditioning,
                                        prompts,
                                        vcn_max_epochs=50,
                                        vcn_optimizer_lr=0.01,
                                        vcn_flow_error_scale=1,
                                        vcn_warp_error_scale=1,
                                        vcn_lineart_error_scale=1,
                                        ):
    """
    Craft an optimal noise to generate temporally consistent video
    args :
    noise : float tensor in [T , H , W , C ] format
    conditioning : float tensor in [T , H , W , C ] format
    unconditional_conditioning : float tensor in [T , H , W , C ] format
    prompts : textual conditioning strings
    """

    print("\n====>vram start", torch.cuda.memory_allocated('cuda') / 1024**3) if vram_debug else None

    self.sd_model.eval()

    vcn_minimal_loss = None

    optimal_latent = init_latent
    optimal_noise = noise

    init_latent.requires_grad_(True)
    noise.requires_grad_(True)

    optimizer_noise = torch.optim.AdamW([noise], lr=vcn_optimizer_lr)
    optimizer_latent = torch.optim.AdamW([init_latent], lr=vcn_optimizer_lr)

    refs = []
    index = 0
    for frame in self.vcn_previous_frames:
        ref = torch.Tensor(np.array(frame)).to('cuda')

        ref_lineart = None
        if vcn_lineart_error_scale > 0:
            ref_lineart = self.get_lineart(ref).to('cuda')

        ref_flow = torch.sum(torch.stack(self.vcn_flows[:index+1]), dim=0).to('cuda')

        refs.append([
            ref,
            ref_flow,
            ref_lineart,
            ])

        index = index+1

    for epoch in range (vcn_max_epochs):

      print("\n====>vram epoch", epoch, torch.cuda.memory_allocated('cuda') / 1024**3) if vram_debug else None

      optimizer_noise.zero_grad ()
      optimizer_latent.zero_grad ()

      with torch.enable_grad():
        try:
            samples_ddim = self.sampler.sample_img2img(self,
              init_latent,
              noise,
              conditioning,
              unconditional_conditioning,
              image_conditioning=self.image_conditioning,
              steps=self.vcn_sample_steps,
              )
        except Exception as e:
            print("\n===>Exception", e)

        x_samples_ddim = [decode_first_stage(self.sd_model, samples_ddim)[0]]
        x_samples_ddim = torch.stack(x_samples_ddim).float()
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

        x_sample = x_samples_ddim[0] * 255.0
        x_sample = x_sample.permute(1, 2, 0)

        print("\n====>vram sampled", epoch, torch.cuda.memory_allocated('cuda') / 1024**3) if vram_debug else None

        loss = []
        index = 0
        for ref in refs:
            print("\n====>vram ref", index, torch.cuda.memory_allocated('cuda') / 1024**3) if vram_debug else None

            err = self.calc_error(
                x_sample,
                ref[0],
                ref[1],
                ref[2],
                vcn_warp_error_scale,
                vcn_flow_error_scale,
                vcn_lineart_error_scale,
                )
            index = index + 1

            print("\n===> ref loss", index, err)

            # normalized by number of non - zero pixels
            loss . append ( err )

        print("\n====>vram loss", torch.cuda.memory_allocated('cuda') / 1024**3) if vram_debug else None

        loss = max(loss)
        self.loss_history.append(loss.item())

        if vcn_minimal_loss == None or loss < vcn_minimal_loss:
          vcn_minimal_loss = loss
          optimal_latent = init_latent.clone()
          optimal_noise = noise

      print("\n====>vram backward", torch.cuda.memory_allocated('cuda') / 1024**3) if vram_debug else None

      loss.backward ()
      optimizer_noise.step()
      optimizer_latent.step()

      current_latent_lr = optimizer_latent.param_groups[0]['lr']
      current_noise_lr = optimizer_noise.param_groups[0]['lr']
      print("\n===> loss", epoch, loss.item(), current_latent_lr, current_noise_lr, hash_tensor(init_latent), hash_tensor(noise))
      print("\n====>vram loop end", torch.cuda.memory_allocated('cuda') / 1024**3) if vram_debug else None

    print("\n====> final", vcn_minimal_loss, hash_tensor(optimal_noise), hash_tensor(optimal_latent))
    print("\n====>vram end", torch.cuda.memory_allocated('cuda') / 1024**3) if vram_debug else None
    return optimal_latent, optimal_noise


def init():
  load_cnet_models()

def infer(controlnets=[],
          sag_enabled=False,
          vcn_flows = [],
          vcn_key_frame = None,
          vcn_previous_frames = [],
          vcn_max_epochs = 50,
          vcn_optimizer_lr = 0.01,
          vcn_scheduler_factor = 0.1,
          vcn_scheduler_patience = 5,
          vcn_sample_steps = 10,
          vcn_fidelity_oriented_compensation = False,
          vcn_fidelity_oriented_compensation_mask = 0.1,
          vcn_adain = False,
          vcn_blur = False,
          vcn_blur_kernel = 9,
          vcn_blur_sigma = 1.0,
          **kwargs):

  print("\n====>vram infer", torch.cuda.memory_allocated('cuda') / 1024**3) if vram_debug else None

  p = StableDiffusionProcessingImg2ImgVCN(
      sd_model=shared.sd_model,
      do_not_save_samples=True,

      vcn_key_frame = vcn_key_frame,
      vcn_flows = vcn_flows,
      vcn_max_epochs = vcn_max_epochs,
      vcn_previous_frames = vcn_previous_frames,
      vcn_optimizer_lr = vcn_optimizer_lr,
      vcn_scheduler_factor = vcn_scheduler_factor,
      vcn_scheduler_patience = vcn_scheduler_patience,
      vcn_sample_steps = vcn_sample_steps,
      vcn_fidelity_oriented_compensation = vcn_fidelity_oriented_compensation,
      vcn_fidelity_oriented_compensation_mask = vcn_fidelity_oriented_compensation_mask,
      vcn_adain = vcn_adain,
      vcn_blur = vcn_blur,
      vcn_blur_kernel = vcn_blur_kernel,
      vcn_blur_sigma = vcn_blur_sigma,

      **kwargs
  )

  args = {}
  script_args = {}

  units = []
  units, cnet_args_from, cnet_args_to = append_cnet_units(units, controlnets, **kwargs)
  units, sag_args_from, sag_args_to = append_sag_units(units, sag_enabled, **kwargs)

  script_args = tuple(units)

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

    elif sag_enabled and script.title() == 'Self Attention Guidance':
      p.scripts.scripts[index].args_from = sag_args_from
      p.scripts.scripts[index].args_to = sag_args_to
      enabled_scripts.append(p.scripts.scripts[index])

  p.scripts.scripts = enabled_scripts
  p.scripts.alwayson_scripts = enabled_scripts

  print("\n====>vram p init", torch.cuda.memory_allocated('cuda') / 1024**3) if vram_debug else None

  shared.state.begin()
  processed = process_images(p)
  shared.state.end()

  print("\n====>vram p processed", torch.cuda.memory_allocated('cuda') / 1024**3) if vram_debug else None

  processed.loss_history = p.loss_history

  p.close()
  shared.total_tqdm.clear()

  print("\n====>vram p close", torch.cuda.memory_allocated('cuda') / 1024**3) if vram_debug else None

  del p
  devices.torch_gc()

  print("\n====>vram p del", torch.cuda.memory_allocated('cuda') / 1024**3) if vram_debug else None

  return processed

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

def get_flow_field(flow,
                   image,
                   arrow_scale = 1,
                   arrow_thickness = 1,
                   ):


  show = image.copy()
  img = ImageDraw.Draw(show)

  # Iterate over each pixel in the flow array
  for y in range(0, flow.shape[0], 10):
      for x in range(0, flow.shape[1], 10):
          # Calculate the flow vector at the current pixel
          flow_x = flow[y, x, 0]  # Flow in x-direction
          flow_y = flow[y, x, 1]  # Flow in y-direction

          # Calculate the arrow endpoints
          arrow_start = (x, y)
          arrow_end = (x + flow_x * arrow_scale, y + flow_y * arrow_scale)

          # Draw the arrow on the image
          img.line([arrow_start, arrow_end], fill='green', width=0)
  return show


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

def encode(image):
  image = image.half() / 255.0
  image = image.permute(0, 3, 1, 2)
  image = 2. * image - 1
  latent = shared.sd_model.get_first_stage_encoding(shared.sd_model.encode_first_stage(image))
  return latent

def decode(latent):
  image = shared.sd_model.decode_first_stage(latent.half())
  image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
  image = image * 255.0
  image = image.permute(0, 2, 3, 1)
  return image

def fidelity_oriented_zeroshot_encoding(x0r, l=0.1):
  """
  2306.07954
  """

  x0rr = encode(decode(x0r))

  corr = x0r - x0rr

  corr = x0r + torch.clamp(corr, min=0.0, max=l)

  return  corr


# AdaIN https://github.com/naoto0804/pytorch-AdaIN
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())

# https://github.com/ashen-sensored/sd_webui_SAG
def gaussian_blur_2d(img, kernel_size, sigma):
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    img = Fnn.pad(img, padding, mode="reflect")
    img = Fnn.conv2d(img, kernel2d, groups=img.shape[-3])

    return img
