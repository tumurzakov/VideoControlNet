
from inspect import isfunction
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
import xformers

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
    def __init__(self,
                 enabled=False,
                 contexts = None,
                 output_attn_start = 3,
                 output_attn_end = 12,
                 input_attn_start = 1,
                 input_attn_end = 9,
                 middle_attn=True,
                 previous_weight = 1,
                 current_weight = 0,
                 ):

        self.enabled = enabled
        self.contexts = contexts
        self.output_attn_start=output_attn_start
        self.output_attn_end=output_attn_end
        self.input_attn_start=input_attn_start
        self.input_attn_end=input_attn_end
        self.middle_attn=middle_attn
        self.previous_weight=previous_weight
        self.current_weight=current_weight


def xattn_forward_cfa(self, x, context=None, mask=None):
    global cfa_previous_contexts, cfa_current_contexts, cfa_index, cfa_previous_weight, cfa_current_weight
    cfa_current_contexts.append(x.detach())

    def cfa_calc_attn(self, x, context=None, mask=None):
        dim_head = int(self.scale ** -2)

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * dim_head)
        )
        return self.to_out(out)

    outc = cfa_calc_attn(self, x)

    current_weight = cfa_current_weight
    previous_weight = cfa_previous_weight

    outp = None
    if cfa_previous_contexts != None and len(cfa_previous_contexts) > cfa_index:
        outp = cfa_calc_attn(self, x, cfa_previous_contexts[cfa_index].to('cuda'))

    else:
        current_weight = 1
        previous_weight = 0

    out = outc * current_weight

    if outp != None:
        out = out + outp * previous_weight

    cfa_index = cfa_index + 1

    return out

saved_original_selfattn_forward = {}
current_selfattn_map = None
cfa_enabled = False

cfa_previous_contexts = None
cfa_current_contexts = []
cfa_index = 0
cfa_layers = 0
cfa_previous_weight = 1
cfa_current_weight = 0

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

        global cfa_enabled, cfa_layers, cfa_previous_contexts, cfa_current_contexts, cfa_index, cfa_previous_weight, cfa_current_weight

        enabled, cfa_previous_contexts = [False, None]

        cfa_unit = None
        for unit in p.script_args:
            if "CFAUnit" in type(unit).__name__:
                cfa_unit = unit
                enabled = unit.enabled
                cfa_previous_contexts = unit.contexts
                cfa_previous_weight = unit.previous_weight
                cfa_current_weight = unit.current_weight

        if enabled:
            print("\n===>CFA enabled")

            p.cfa_enabled = True
            cfa_enabled = True
            cfa_index = 0
            cfa_layers = 0
            cfa_current_contexts = []
            global saved_original_selfattn_forward
            # replace target self attention module in unet with ours

            for i in range(cfa_unit.input_attn_start,cfa_unit.input_attn_end):
                if '1' in shared.sd_model.model.diffusion_model.input_blocks[i]._modules:
                    print("\n===>CFA replace input", i)
                    org_attn_module = shared.sd_model.model.diffusion_model.input_blocks[i]._modules['1'].transformer_blocks._modules['0'].attn1
                    saved_original_selfattn_forward['input_%d' % i] = org_attn_module.forward
                    org_attn_module.forward = xattn_forward_cfa.__get__(org_attn_module,org_attn_module.__class__)
                    cfa_layers = cfa_layers + 1

            if cfa_unit.middle_attn:
                print("\n===>CFA replace middle")
                org_attn_module = shared.sd_model.model.diffusion_model.middle_block._modules['1'].transformer_blocks._modules['0'].attn1
                saved_original_selfattn_forward['middle'] = org_attn_module.forward
                org_attn_module.forward = xattn_forward_cfa.__get__(org_attn_module,org_attn_module.__class__)
                cfa_layers = cfa_layers + 1

            for i in range(cfa_unit.output_attn_start,cfa_unit.output_attn_end):
                if '1' in shared.sd_model.model.diffusion_model.output_blocks[i]._modules:
                    print("\n===>CFA replace output", i)
                    org_attn_module = shared.sd_model.model.diffusion_model.output_blocks[i]._modules['1'].transformer_blocks._modules['0'].attn1
                    saved_original_selfattn_forward['output_%d' % i] = org_attn_module.forward
                    org_attn_module.forward = xattn_forward_cfa.__get__(org_attn_module,org_attn_module.__class__)
                    cfa_layers = cfa_layers + 1
        else:
            cfa_enabled = False

        return

    def postprocess(self, p, processed, *args):
        enabled = [False]

        if len(args) == 1:
            enabled = args[0]

        cfa_unit = None
        for unit in p.script_args:
            if "CFAUnit" in type(unit).__name__:
                cfa_unit = unit
                enabled = unit.enabled

        if enabled:
            # restore original self attention module forward function
            for i in range(cfa_unit.input_attn_start,cfa_unit.input_attn_end):
                if '1' in shared.sd_model.model.diffusion_model.input_blocks[i]._modules:
                    print("\n===>CFA restore input", i)
                    attn_module = shared.sd_model.model.diffusion_model.input_blocks[i]._modules['1'].transformer_blocks._modules['0'].attn1
                    attn_module.forward = saved_original_selfattn_forward['input_%d' % i]

            if cfa_unit.middle_attn:
                print("\n===>CFA restore middle")
                attn_module = shared.sd_model.model.diffusion_model.middle_block._modules['1'].transformer_blocks._modules['0'].attn1
                attn_module.forward = saved_original_selfattn_forward['middle']

            for i in range(cfa_unit.output_attn_start,cfa_unit.output_attn_end):
                if '1' in shared.sd_model.model.diffusion_model.output_blocks[i]._modules:
                    print("\n===>CFA restore output", i)
                    attn_module = shared.sd_model.model.diffusion_model.output_blocks[i]._modules['1'].transformer_blocks._modules['0'].attn1
                    attn_module.forward = saved_original_selfattn_forward['output_%d' % i]

            global cfa_layers, cfa_current_contexts
            processed.cfa_layers = cfa_layers
            if len(cfa_current_contexts) > 0:
                processed.cfa_contexts = cfa_current_contexts
            cfa_current_contexts = []
            cfa_index = 0
        return
