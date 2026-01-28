import torch
import math
import numpy as np
import torch.nn as nn
import copy
from einops import rearrange
from torch.nn.modules.module import T
from transformers.cache_utils import DynamicCache

from tqdm import tqdm
from transformers import Qwen2ForCausalLM, Qwen2Config, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.autograd.function import Function
from .diffusion_utils import *
from .gaussian_diffusion import *
from .respace import *
from .misc import *
from .diffloss import *
from torch.utils.checkpoint import checkpoint

from .configuration_harmon import HarmonConfig
from .vae import AutoencoderKL
from .mar import mar_base, mar_large, mar_huge


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),)


def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len, device=order.device)
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()],
                            src=torch.ones(bsz, seq_len, device=order.device)).bool()
    return masking

class _ScaleGradient(Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None

class HarmonModel(PreTrainedModel):
    config_class = HarmonConfig

    def __init__(self, config: HarmonConfig):
        super().__init__(config)
        self.grad_scale = 0.1
        self.IMAGE_TOKEN_INDEX = None
        # VAE
        self.vae = AutoencoderKL(
            embed_dim=16,
            ch_mult=(1, 1, 2, 2, 4)
        )
        self.vae_scale = 0.2325

        # LLM
        self.llm = Qwen2ForCausalLM(config=Qwen2Config.from_dict(config.llm))

        # MAR
        mar_config = copy.deepcopy(config.mar)
        mar_type = mar_config.pop('type')
        if mar_type == 'mar_base':
            self.mar = mar_base(**mar_config)
        elif mar_type == 'mar_large':
            self.mar = mar_large(**mar_config)
        elif mar_type == 'mar_huge':
            self.mar = mar_huge(**mar_config)
        else:
            raise ValueError

        # projection layers
        self.proj_in = build_mlp(hidden_size=self.mar.encoder_embed_dim,
                                 projector_dim=self.llm.config.hidden_size,
                                 z_dim=self.llm.config.hidden_size)
        self.proj_out = build_mlp(hidden_size=self.llm.config.hidden_size,
                                  projector_dim=self.llm.config.hidden_size,
                                  z_dim=self.mar.encoder_embed_dim)

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.mar.gradient_checkpointing_disable()

    def gradient_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        self.mar.gradient_checkpointing_enable()

    @property
    def llm_model(self):
        return self.llm.model

    @property
    def device(self):
        return self.llm.device

    @property
    def dtype(self):
        return self.llm.dtype

    @property
    def gen_seq_len(self):
        return self.mar.seq_len

    @property
    def token_embed_dim(self):
        return self.vae.embed_dim * (self.mar.patch_size ** 2)

    @torch.no_grad()
    def encode(self, x):
        posterior = self.vae.encode(x)
        z = posterior.mode().mul_(self.vae_scale)
        z = rearrange(z, 'b c (m p) (n q) -> b m n (c p q)',
                      p=self.mar.patch_size, q=self.mar.patch_size)

        return z

    def encode_train(self, x):
        def run_vae_encoder(x_in):
            posterior = self.vae.encode(x_in)
            return posterior.mode()

        z = checkpoint(run_vae_encoder, x, use_reentrant=False)
        z = z * self.vae_scale
        z = rearrange(z, 'b c (m p) (n q) -> b m n (c p q)',
                      p=self.mar.patch_size, q=self.mar.patch_size)

        return z

    @torch.no_grad()
    def decode(self, z):
        z /= self.vae_scale
        z = rearrange(z, 'b m n (c p q) -> b c (m p) (n q)',
                      p=self.mar.patch_size, q=self.mar.patch_size)

        x = self.vae.decode(z)
        return x

    def decode_train(self, z):
        z /= self.vae_scale
        z = rearrange(z, 'b m n (c p q) -> b c (m p) (n q)',
                      p=self.mar.patch_size, q=self.mar.patch_size)

        def run_vae_decoder(x):
            return self.vae.decode(x)

        out = checkpoint(run_vae_decoder, z, use_reentrant=False)

        return out

    def prepare_forward_input(self,
                              x,
                              inputs_embeds=None,
                              input_ids=None,
                              attention_mask=None,
                              past_key_values=None):
        b, l, _ = x.shape
        attention_mask = attention_mask.to(device=self.device, dtype=torch.bool)
        attention_mask = torch.cat([
            attention_mask, attention_mask.new_ones(b, l)
        ], dim=1)
        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        position_ids[position_ids < 0] = 0

        # import pdb; pdb.set_trace()

        # prepare context
        if past_key_values is not None:
            inputs_embeds = x
            position_ids = position_ids[:, -l:]
        else:
            if inputs_embeds is None:
                input_ids = input_ids.to(self.device)
                inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([inputs_embeds, x], dim=1)

        return dict(inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values)

    def extract_visual_feature(self, x, mask=None, detach=False):
        b, m, n, _ = x.shape
        x = x.view(b, m*n, -1)
        # x: b mn c
        if mask is None:
            mask = torch.zeros_like(x[..., 0])
        null_embeds = self.mar.fake_latent.expand(x.shape[0], -1)
        x_enc = self.mar.forward_mae_encoder(x, mask, null_embeds, image_shape=(m, n))

        z_enc = self.proj_in(x_enc)
        # Move buffers to the end of the image sequence
        z_enc = torch.cat([
            z_enc[:, self.mar.buffer_size:],
            z_enc[:, :self.mar.buffer_size]], dim=1)

        if detach:
            x_enc = x_enc.detach()
            z_enc = z_enc.detach()

        return x_enc, z_enc

    def forward_mae_encoder(self, x, mask, detach=False, **context):
        b, m, n, _ = x.shape
        x_enc, z_enc = self.extract_visual_feature(x, mask=mask, detach=detach)
        inputs = self.prepare_forward_input(x=z_enc, **context)
        output = self.llm_model(**inputs, return_dict=True)

        z_llm = output.last_hidden_state[:, -z_enc.shape[1]:]

        # move buffers back to the start of the image sequence
        z_llm = torch.cat([
            z_llm[:, -self.mar.buffer_size:],
            z_llm[:, :-self.mar.buffer_size]], dim=1)

        # residual learning
        x_enc = x_enc + self.proj_out(z_llm)

        return x_enc

    @staticmethod
    def curtail_cache(past_key_values, cur_len):
        for past_key_values_ in past_key_values:
            keys, values = past_key_values_
            keys.data = keys.data[:, :, :cur_len]
            values.data = values.data[:, :, :cur_len]

    @torch.no_grad()
    def sample1(self,
               input_ids=None, inputs_embeds=None,
               attention_mask=None, num_iter=64, cfg=1.0, cfg_schedule="constant", temperature=1.0,
               progress=False, mask=None, past_key_values=None, image_shape=None, x_con=None, **kwargs):
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        bsz = attention_mask.shape[0]
        if cfg != 1.0:
            assert bsz % 2 == 0

        if image_shape is None:
            m = n = int(self.gen_seq_len ** 0.5)
        else:
            m, n = image_shape

        if mask is None:
            mask = torch.ones(bsz, m * n, device=self.device, dtype=self.dtype)
        else:
            mask = mask.view(bsz, m * n)
        tokens = torch.zeros(bsz, m * n, self.token_embed_dim,
                             device=self.device, dtype=self.dtype)
        orders = self.mar.sample_orders(bsz, seq_len=m * n)
        if cfg != 1.0:
            orders[bsz // 2:] = orders[:bsz // 2]

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)

        # past key values can be prepared outside (usually in multi-turn editing)
        if past_key_values is None:
            output = self.llm_model(inputs_embeds=inputs_embeds,
                                    attention_mask=None,
                                    position_ids=None,
                                    past_key_values=DynamicCache.from_legacy_cache(),
                                    return_dict=True,
                                    use_cache=True)
            past_key_values = output.past_key_values

        # generate latents
        for step in indices:
            cur_tokens = tokens.clone()
            x_enc = self.forward_mae_encoder(tokens.view(bsz, m, n, -1),
                                             mask.to(self.dtype),
                                             past_key_values=past_key_values,
                                             # inputs_embeds=inputs_embeds,
                                             attention_mask=attention_mask)
            # import pdb; pdb.set_trace()
            self.curtail_cache(past_key_values, inputs_embeds.shape[1])
            # import pdb; pdb.set_trace()

            z = self.mar.forward_mae_decoder(x_enc, mask.to(self.dtype), image_shape=(m, n), x_con=x_con)

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(m * n * mask_ratio)]).to(self.device)

            # masks out at least one for the next iteration
            mask_len = torch.maximum(torch.Tensor([1]).to(self.device),
                                     torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, bsz, m * n).to(self.device)
            if cfg != 1.0:
                mask_next[bsz // 2:] = mask_next[:bsz // 2]
            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask = mask_next
            # if not cfg == 1.0:
            #     mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            # sample token latents for this step
            z = z[mask_to_pred.nonzero(as_tuple=True)]
            # cfg schedule follow Muse
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (m * n - mask_len[0]) / (m * n)
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError
            sampled_token_latent = self.mar.diffloss.sample(z, temperature, cfg_iter).to(self.dtype)

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            if cfg != 1.0:
                cur_tokens[bsz // 2:] = cur_tokens[:bsz // 2]
            tokens = cur_tokens.clone()

        pred = self.decode(tokens.view(bsz, m, n, -1))

        if cfg != 1.0:
            pred = pred[:bsz // 2]
        return pred

    def sample(self,
               input_ids=None, inputs_embeds=None,
               attention_mask=None, num_iter=64, cfg=1.0, cfg_schedule="constant", temperature=1.0,
               progress=False, mask=None, past_key_values=None, image_shape=None, x_con=None, num_timesteps=None,
               **kwargs):

        # 1. 基础准备
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        bsz = attention_mask.shape[0]
        if cfg != 1.0:
            assert bsz % 2 == 0
        if image_shape is None:
            m = n = int(self.gen_seq_len ** 0.5)
        else:
            m, n = image_shape

        # 初始化 Mask (确保是 dtype 类型，例如 float/bfloat16)
        if mask is None:
            mask = torch.ones(bsz, m * n, device=self.device, dtype=self.dtype)
        else:
            mask = mask.view(bsz, m * n)

        # Tokens 初始化 (Phase 1 不需要梯度)
        tokens = torch.zeros(bsz, m * n, self.token_embed_dim,
                             device=self.device, dtype=self.dtype, requires_grad=False)

        orders = self.mar.sample_orders(bsz, seq_len=m * n)
        if cfg != 1.0:
            orders[bsz // 2:] = orders[:bsz // 2]
        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)

        # 预计算 Prompt Cache
        if past_key_values is None:
            with torch.no_grad():
                output = self.llm_model(inputs_embeds=inputs_embeds,
                                        attention_mask=None,
                                        position_ids=None,
                                        past_key_values=DynamicCache.from_legacy_cache(),
                                        return_dict=True,
                                        use_cache=True)
                past_key_values = output.past_key_values

        # ==========================================================
        # Phase 1: 完整采样 (Rollout) - 全程无梯度
        # ==========================================================
        trajectory = []

        with torch.no_grad():
            for step in indices:
                is_last_step = (step == num_iter - 1)

                cur_tokens = tokens.clone()

                # 记录状态
                step_input_snapshot = {
                    'tokens': cur_tokens.clone(),
                    'mask': mask.clone(),
                    'step': step
                }

                # --- 1. Encoder ---
                x_enc = self.forward_mae_encoder(
                    cur_tokens.view(bsz, m, n, -1),
                    mask.to(self.dtype),  # <--- 强制转换
                    past_key_values=past_key_values,
                    attention_mask=attention_mask
                )
                self.curtail_cache(past_key_values, inputs_embeds.shape[1])

                # --- 2. Decoder ---
                z = self.mar.forward_mae_decoder(x_enc, mask.to(self.dtype), image_shape=(m, n), x_con=x_con)

                # --- 3. Mask Schedule ---
                mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
                mask_len = torch.Tensor([np.floor(m * n * mask_ratio)]).to(self.device)
                mask_len = torch.maximum(torch.Tensor([1]).to(self.device),
                                         torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

                # 生成下一步 mask (可能是 bool 或 float，取决于 mask_by_order 实现)
                mask_next = mask_by_order(mask_len[0], orders, bsz, m * n).to(self.device)


                if cfg != 1.0: mask_next[bsz // 2:] = mask_next[:bsz // 2]

                # mask_to_pred 需要是 bool 用于索引
                mask_to_pred = mask[:bsz].bool() if is_last_step else torch.logical_xor(mask[:bsz].bool(),
                                                                                        mask_next.bool())

                # --- 4. Diffusion Sample ---
                z_sel = z[mask_to_pred.nonzero(as_tuple=True)]
                if cfg_schedule == "linear":
                    cfg_iter = 1 + (cfg - 1) * (m * n - mask_len[0]) / (m * n)
                else:
                    cfg_iter = cfg

                sampled_latent = self.mar.diffloss.sample(z_sel, temperature, cfg_iter, num_timesteps).to(
                    self.dtype)

                # 记录 Target
                step_input_snapshot['target_latent'] = sampled_latent.clone()
                step_input_snapshot['mask_to_pred'] = mask_to_pred
                step_input_snapshot['cfg_iter'] = cfg_iter
                trajectory.append(step_input_snapshot)

                # 5. 更新 tokens 和 mask
                mask = mask_next  # 此时 mask 已经是 float (dtype) 了
                cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_latent
                if cfg != 1.0:
                    cur_tokens[bsz // 2:] = cur_tokens[:bsz // 2]
                tokens = cur_tokens

            # 生成最终图片
            pred_image = self.decode(tokens.view(bsz, m, n, -1))
            if cfg != 1.0: pred_image = pred_image[:bsz // 2]

        # ==========================================================
        # Phase 2: 随机单步训练 (Random Step Training) - 开启梯度
        # ==========================================================
        import random
        target_step_idx = random.randint(0, num_iter - 1)
        target_data = trajectory[target_step_idx]

        curr_tokens = target_data['tokens'].detach()
        curr_mask = target_data['mask']
        target_action = target_data['target_latent'].detach()

        with torch.enable_grad():
            # --- A. Encoder ---
            x_enc = self.forward_mae_encoder(
                curr_tokens.view(bsz, m, n, -1),
                curr_mask.to(self.dtype),  # <--- 强制转换
                past_key_values=past_key_values,
                attention_mask=attention_mask
            )

            # --- B. Decoder ---
            # 【修复点 3】: 强制 curr_mask 转为 dtype
            z = self.mar.forward_mae_decoder(x_enc, curr_mask.to(self.dtype), image_shape=(m, n), x_con=x_con)

            # --- C. Diffusion Sample ---
            mask_to_pred = target_data['mask_to_pred']  # 这是 bool，用来切片没问题
            indices_tuple = mask_to_pred.nonzero(as_tuple=True)
            batch_indices = indices_tuple[0]  # [Total_Active_Tokens]，记录每个 token 属于哪个 batch
            z_sel = z[mask_to_pred.nonzero(as_tuple=True)]

            if cfg != 1.0:
                orig_bsz = bsz // 2
                # batch_indices 可能是 cuda tensor，保证同 device 与 long dtype
                batch_indices = batch_indices.to(self.device).long()
                # map to original batch ids
                batch_indices = batch_indices % orig_bsz

            step_pred_latent = self.mar.diffloss.sample_refl(
                z_sel,
                temperature,
                target_data['cfg_iter'],
                num_timesteps
            ).to(self.dtype)

        return pred_image, step_pred_latent, target_action, batch_indices

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            pixel_values=None,
            pixel_value_refs=None,
            labels=None,
            image_grid_thw=None,
            task="image2text",  # "text2image" or "image2text"
            return_dict=True,
    ):
        loss, logits = None, None

        if task == "text2image":
            # ===== Text -> Image (保持不变) =====
            x = pixel_values.to(dtype=self.dtype, device=self.device)
            x = self.encode(x)  # b m n c
            b, m, n, _ = x.shape
            gt_latents = x.clone().detach().view(b, m * n, -1)

            orders = self.mar.sample_orders(bsz=b, seq_len=m * n)
            mask = self.mar.random_masking(x.flatten(1, 2), orders)

            x_enc = self.forward_mae_encoder(
                x, mask, input_ids=input_ids, attention_mask=attention_mask
            )
            z = self.mar.forward_mae_decoder(x_enc, mask, image_shape=(m, n))

            loss = self.mar.forward_loss(z=z, target=gt_latents, mask=mask)
            logits = None

        else:  # ===== Image -> Text =====
            # 初始化 loss_null，确保在所有分支都有定义
            loss_null = 0.0

            # 1. 纯文本情况 (pixel_values is None)
            if pixel_values is None:
                inputs_embeds = self.llm.get_input_embeddings()(input_ids)

            # 2. 有图片情况
            else:
                x = pixel_values.to(dtype=self.dtype, device=self.device)
                x = self.encode(x)  # b m n c
                _, z_enc = self.extract_visual_feature(x)

                if self.grad_scale is not None:
                    z_enc = _ScaleGradient.apply(z_enc, self.grad_scale)

                # 处理参考图 (可能为 None)
                z_enc_ref = None
                if pixel_value_refs is not None:
                    x_ref = pixel_value_refs.to(dtype=self.dtype, device=self.device)
                    x_ref = self.encode(x_ref)  # b m n c
                    _, z_enc_ref = self.extract_visual_feature(x_ref)

                    if self.grad_scale is not None:
                        z_enc_ref = _ScaleGradient.apply(z_enc_ref, self.grad_scale)

                inputs_embeds = z_enc.new_zeros(*input_ids.shape, self.llm.config.hidden_size)

                # 填充文本 Embeddings
                text_mask = (input_ids != self.IMAGE_TOKEN_INDEX)
                inputs_embeds[text_mask] = self.llm.get_input_embeddings()(input_ids[text_mask])

                # 填充图像 Embeddings
                for b in range(input_ids.size(0)):
                    img_pos = (input_ids[b] == self.IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]

                    n1 = z_enc[b].size(0)
                    n2 = z_enc_ref[b].size(0) if z_enc_ref is not None else 0

                    if len(img_pos) < n1 + n2:
                        raise ValueError(
                            f"batch {b} IMAGE_TOKEN tokens ({len(img_pos)}) is less than required features ({n1 + n2})")

                    # 填入第一张图
                    inputs_embeds[b, img_pos[:n1]] = z_enc[b]
                    # 填入参考图 (如果有)
                    if z_enc_ref is not None:
                        inputs_embeds[b, img_pos[n1:n1 + n2]] = z_enc_ref[b]

            # LLM Forward
            output = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True
            )

            last_hidden_state = output.last_hidden_state
            logits = self.llm.get_output_embeddings()(last_hidden_state)

            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss_i2t = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                # 加上 loss_null 确保梯度图连通
                loss = loss_i2t + loss_null

        if not return_dict:
            return (loss, logits)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

    def generate(
            self,
            input_ids=None,
            attention_mask=None,
            pixel_values=None,
            pixel_value_refs=None,
            image_grid_thw=None,
            generation_config=None,
            **kwargs
    ):
        """
        Harmon 的 generate 接口，封装视觉处理后调用 self.llm.generate。
        """
        # 1. 纯文本生成
        if pixel_values is None:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        # 2. 带图生成
        else:
            x = pixel_values.to(dtype=self.dtype, device=self.device)
            x = self.encode(x)  # b m n c
            _, z_enc = self.extract_visual_feature(x)

            if self.grad_scale is not None:
                z_enc = _ScaleGradient.apply(z_enc, self.grad_scale)

            # 处理参考图 (可能为 None)
            z_enc_ref = None
            if pixel_value_refs is not None:
                x_ref = pixel_value_refs.to(dtype=self.dtype, device=self.device)
                x_ref = self.encode(x_ref)
                _, z_enc_ref = self.extract_visual_feature(x_ref)

                if self.grad_scale is not None:
                    z_enc_ref = _ScaleGradient.apply(z_enc_ref, self.grad_scale)

            # 构建 Embeddings
            inputs_embeds = z_enc.new_zeros(*input_ids.shape, self.llm.config.hidden_size)

            # 填入文本
            text_mask = (input_ids != self.IMAGE_TOKEN_INDEX)
            inputs_embeds[text_mask] = self.llm.get_input_embeddings()(input_ids[text_mask])

            # 填入图像
            for b in range(input_ids.size(0)):
                img_pos = (input_ids[b] == self.IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]

                n1 = z_enc[b].size(0)
                n2 = z_enc_ref[b].size(0) if z_enc_ref is not None else 0

                if len(img_pos) < n1 + n2:
                    raise ValueError(f"batch {b} IMAGE_TOKEN 不足")

                inputs_embeds[b, img_pos[:n1]] = z_enc[b]
                if z_enc_ref is not None:
                    inputs_embeds[b, img_pos[n1:n1 + n2]] = z_enc_ref[b]

        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            **kwargs
        )
