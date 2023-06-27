
from inspect import isfunction
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

from modules.processing import StableDiffusionProcessing

import math

import modules.scripts as scripts
from modules import shared
import gradio as gr

from modules.script_callbacks import on_cfg_denoiser,CFGDenoiserParams, CFGDenoisedParams, on_cfg_denoised, AfterCFGCallbackParams, on_cfg_after_cfg

import os

_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")
def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class CFAUnit:
    def __init__(self, enabled=False, contexts = None):
        self.enabled = enabled
        self.contexts = contexts

def xattn_forward_log(self, x, context=None, mask=None):
    h = self.heads

    global cfa_previous_contexts, cfa_current_contexts
    cfa_current_contexts.append(x)

    q = self.to_q(x)

    if cfa_previous_contexts != None:
        context = default(cfa_previous_contexts[cfa_index], x)

    context = default(context, x)
    k = self.to_k(context)
    v = self.to_v(context)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    # force cast to fp32 to avoid overflowing
    if _ATTN_PRECISION == "fp32":
        with torch.autocast(enabled=False, device_type='cuda'):
            q, k = q.float(), k.float()
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
    else:
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

    del q, k

    if exists(mask):
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    sim = sim.softmax(dim=-1)

    self.attn_probs = sim
    global current_selfattn_map
    current_selfattn_map = sim

    out = einsum('b i j, b j d -> b i d', sim, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    out = self.to_out(out)
    global current_outsize
    current_outsize = out.shape[-2:]

    cfa_index = cfa_index + 1
    return out

saved_original_selfattn_forward = None
current_selfattn_map = None
cfa_enabled = False

cfa_previous_contexts = None
cfa_current_contexts = []
cfa_index = 0

current_xin = None
current_outsize = (64,64)
current_batch_size = 1
current_degraded_pred= None
current_unet_kwargs = {}
current_uncond_pred = None
current_degraded_pred_compensation = None

class Script(scripts.Script):

    def __init__(self):
        pass

    def title(self):
        return "Cross Frame Attention"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion('Cross Frame Attention', open=False):
            enabled = gr.Checkbox(label="Enabled", default=False)

        return [enabled]

    def process(self, p: StableDiffusionProcessing, *args, **kwargs):

        try:
            if p.cfa_enabled != None:
                return
        except:
            pass

        global cfa_enabled, cfa_previous_contexts

        enabled, cfa_previous_contexts = [False, None]

        for unit in p.script_args:
            if "CFAUnit" in type(unit).__name__:
                enabled = unit.enabled
                cfa_previous_contexts = unit.contexts

        if enabled:
            print("\n===>CFA enabled")

            p.cfa_enabled = True
            cfa_enabled = True
            cfa_index = 0
            global saved_original_selfattn_forward
            # replace target self attention module in unet with ours

            org_attn_module = shared.sd_model.model.diffusion_model.middle_block._modules['1'].transformer_blocks._modules['0'].attn1
            saved_original_selfattn_forward = org_attn_module.forward
            org_attn_module.forward = xattn_forward_log.__get__(org_attn_module,org_attn_module.__class__)
        else:
            cfa_enabled = False

        return

    def postprocess(self, p, processed, *args):
        enabled = [False]

        if len(args) == 1:
            enabled = args[0]

        for unit in p.script_args:
            if "CFAUnit" in type(unit).__name__:
                enabled = unit.enabled

        if enabled:
            # restore original self attention module forward function
            attn_module = shared.sd_model.model.diffusion_model.middle_block._modules['1'].transformer_blocks._modules[
                '0'].attn1
            attn_module.forward = saved_original_selfattn_forward

            global cfa_current_contexts
            processed.cfa_contexts = cfa_current_contexts
        return
