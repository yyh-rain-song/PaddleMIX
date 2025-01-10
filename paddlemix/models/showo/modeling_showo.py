# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn.functional as F

from ppdiffusers.configuration_utils import ConfigMixin
from ppdiffusers.models.modeling_utils import ModelMixin

from .phi import PhiForCausalLM
from .sampling import cosine_schedule, mask_by_random_topk


class Showo(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        w_clip_vit,
        vocab_size,
        llm_vocab_size,
        llm_model_path="",
        codebook_size=8192,
        num_vq_tokens=256,
        load_from_showo=True,
        **kwargs,
    ):
        super().__init__()
        self.w_clip_vit = w_clip_vit
        self.vocab_size = vocab_size
        self.register_to_config(mask_token_id=vocab_size - 1)
        if load_from_showo:
            from .phi import PhiConfig

            config = PhiConfig.from_pretrained(llm_model_path,**kwargs)
            self.showo = PhiForCausalLM(config).to(dtype=kwargs['dtype'])
        else:
            self.showo = PhiForCausalLM.from_pretrained(llm_model_path,**kwargs)

        self.showo.resize_token_embeddings(self.vocab_size)
        self.output_size = self.vocab_size

        if self.w_clip_vit:
            self.mm_projector = paddle.nn.Sequential(
                paddle.nn.Linear(1024, 2048), paddle.nn.GELU(), paddle.nn.Linear(2048, 2048)
            )

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True

    def forward(
        self,
        input_ids,
        input_embeddings=None,
        attention_mask=None,
        labels=None,
        label_smoothing=0.0,
        batch_size_t2i=0,
        batch_size_lm=0,
        batch_size_mmu=0,
        max_seq_length=128,
        labels_mask_text=None,
        labels_mask_image=None,
        **kwargs,
    ):

        if input_embeddings is None:
            logits = self.showo(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)["logits"]
        else:
            logits = self.showo(inputs_embeds=input_embeddings, attention_mask=attention_mask, return_dict=True)[
                "logits"
            ]

        if labels is not None:
            # 1. Mask token prediction (discrete diffusion) for image generation
            # Note that, max_seq_length indicates the maximum number of text tokens, maybe a bit confused.
            loss_t2i = F.cross_entropy(
                logits[:batch_size_t2i, max_seq_length + 1 :].reshape([-1, self.output_size]),
                labels[:batch_size_t2i, max_seq_length + 1 :].reshape([-1]),
                ignore_index=-100,
            )

            # 2. Next token prediction for language modeling
            loss_lm = F.cross_entropy(
                logits[batch_size_t2i : batch_size_t2i + batch_size_lm, :-1].reshape([-1, self.output_size]),
                labels[batch_size_t2i : batch_size_t2i + batch_size_lm, 1:].reshape([-1]),
                ignore_index=-100,
            )

            # 3. Next token prediction for captioning/multimodal understanding
            loss_mmu = F.cross_entropy(
                logits[-batch_size_mmu:, :-1].reshape([-1, self.output_size]),
                labels[-batch_size_mmu:, 1:].reshape([-1]),
                ignore_index=-100,
            )

            return logits, loss_t2i, loss_lm, loss_mmu

        return logits

    def t2i_generate(
        self,
        input_ids: paddle.Tensor = None,
        uncond_input_ids: paddle.Tensor = None,
        attention_mask=None,
        temperature=1.0,
        timesteps=18,  # ideal number of steps is 18 in maskgit paper
        guidance_scale=0,
        noise_schedule=cosine_schedule,
        generator: paddle.Generator = None,
        config=None,
        **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """
        # begin with all image token ids masked
        mask_token_id = self.config.mask_token_id
        num_vq_tokens = config.model.showo.num_vq_tokens
        num_new_special_tokens = config.model.showo.num_new_special_tokens

        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1) : -1].clone()
        input_ids_minus_lm_vocab_size = paddle.where(
            input_ids_minus_lm_vocab_size == mask_token_id,
            mask_token_id,
            input_ids_minus_lm_vocab_size - config.model.showo.llm_vocab_size - num_new_special_tokens,
        )

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, : config.dataset.preprocessing.max_seq_length + 1]

        for step in range(timesteps):
            # for step in range(5):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = paddle.concat(
                    [uncond_prefix, input_ids[:, config.dataset.preprocessing.max_seq_length + 1 :]], axis=1
                )
                model_input = paddle.concat([input_ids, uncond_input_ids])
                cond_logits, uncond_logits = self(model_input, attention_mask=attention_mask).chunk(2)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[
                    :, -(num_vq_tokens + 1) : -1, config.model.showo.llm_vocab_size + num_new_special_tokens : -1
                ]
            else:
                logits = self(input_ids, attention_mask=attention_mask)
                logits = logits[
                    :, -(num_vq_tokens + 1) : -1, config.model.showo.llm_vocab_size + num_new_special_tokens : -1
                ]

            probs = F.softmax(logits, axis=-1)
            sampled = probs.reshape([-1, logits.shape[-1]])

            # sampled_ids = paddle.multinomial(sampled, 1, generator=generator)[:, 0].reshape([*logits.shape[:-1]])
            sampled_ids = paddle.multinomial(sampled, num_samples=1)[:, 0].reshape([*logits.shape[:-1]])

            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            sampled_ids = paddle.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(paddle.to_tensor(ratio))
            # Computes the probabilities of each selected tokens.
            selected_probs = paddle.take_along_axis(
                probs, axis=-1, indices=sampled_ids.astype(dtype="int64")[..., None]
            )
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = paddle.where(unknown_map, selected_probs, paddle.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = paddle.to_tensor([(num_vq_tokens * mask_ratio).floor()]).unsqueeze(0).astype(dtype="int64")
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = paddle.maximum(
                paddle.to_tensor([1]), paddle.minimum(unknown_map.sum(axis=-1, keepdim=True) - 1, mask_len)
            )
            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio)
            # masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature)
            # Masks tokens with lower confidence.
            input_ids[:, -(num_vq_tokens + 1) : -1] = paddle.where(
                masking, mask_token_id, sampled_ids + config.model.showo.llm_vocab_size + num_new_special_tokens
            )
            input_ids_minus_lm_vocab_size = paddle.where(masking, mask_token_id, sampled_ids)

        return sampled_ids

    @paddle.no_grad()
    def mmu_generate(
        self,
        idx=None,
        input_embeddings=None,
        attention_mask=None,
        max_new_tokens=100,
        temperature=1.0,
        top_k=None,
        eot_token=None,
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        result = []
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            # idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            # logits, _ = self(idx_cond)
            logits = self(idx, input_embeddings=input_embeddings, attention_mask=attention_mask)
            L = attention_mask.shape[-1]
            attention_mask = attention_mask.squeeze()
            attention_mask_a = paddle.hstack(
                [
                    attention_mask,  # L, L
                    paddle.zeros((L, 1)) + paddle.finfo(logits.dtype).min,
                ]
            )
            attention_mask_b = paddle.vstack(
                [
                    attention_mask_a,  # L, L+1
                    paddle.hstack([attention_mask[-1, :], paddle.to_tensor([0], dtype=paddle.float32)]).unsqueeze(0),
                ]
            )
            attention_mask = attention_mask_b[None, None, ...]

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = paddle.topk(logits, min(top_k, logits.shape[-1]))
                # logits[logits < v[:, [-1]]] = -float('Inf')
                logits[logits < v] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits.cast("float32"), axis=-1)
            # sample from the distribution
            idx_next = paddle.multinomial(probs, num_samples=1)
            result.append(idx_next[0][0])
            # append sampled index to the running sequence and continue
            if 1:  # self.config.w_clip_vit:
                idx_next_embeddings = self.showo.model.embed_tokens(idx_next)
                input_embeddings = paddle.concat([input_embeddings, idx_next_embeddings], axis=1)
            else:
                idx = paddle.concat((idx, idx_next), axis=1)

            if eot_token is not None and idx_next.cpu() == eot_token:
                break

        return result