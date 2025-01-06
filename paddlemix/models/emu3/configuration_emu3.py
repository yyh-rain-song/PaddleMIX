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
""" Emu3 model configuration"""

from typing import List, Optional

from paddlenlp.transformers.configuration_utils import PretrainedConfig


class Emu3VisionVQConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`Emu3VisionVQ`]. It is used to instantiate an video movq
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a configuration to the VQ model presented in Emu3 paper.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        codebook_size (`int`, *optional*, defaults to 32768):
            Codebook size of the VQ model.
        embed_dim (`int`, *optional*, defaults to 4):
            Dimension of the quantized vector in codebook.
        z_channels (`int`, *optional*, defaults to 4):
            Dimension of the output channel of encoder and the input channel of decoder
        double_z (`bool`, *optional*, defaults to False):
            Whether double the output dim of the encoder.
        in_channels (`int`, *optional*, defaults to 3):
            Input channel of encoder.
        out_channels (`int`, *optional*, defaults to 3):
            Output channel of decoder.
        temporal_downsample_factor (`int`, *optional*, defaults to 4):
            Temporal downsample factor.
        ch (`int`, *optional*, defaults to 256):
            Basic channel number of the intermediate blocks.
        ch_mult (`List[int]`, *optional*, defaults to `[1, 2, 2, 4]`):
            Channel scaling factor of the intermediate blocks.
        num_res_blocks (`int`, *optional*, defaults to 2):
            Residual block number in each stage.
        attn_resolutions (`List[int]`, *optional*, defaults to 3):
            Stage indices to apply attention.
        dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability.

    ```python
    >>> from transformers import Emu3VisionVQ, Emu3VisionVQConfig

    >>> # Initializing a video VQ model of Emu3 configuration
    >>> configuration = Emu3VisionVQConfig()

    >>> # Initializing a model from the Emu3 VQ model style configuration
    >>> model = Emu3VisionVQModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "Emu3VisionVQ"

    def __init__(
        self,
        codebook_size: int = 32768,
        embed_dim: int = 4,
        z_channels: int = 4,
        double_z: bool = False,
        in_channels: int = 3,
        out_channels: int = 3,
        temporal_downsample_factor: int = 4,
        ch: int = 256,
        ch_mult: List[int] = [1, 2, 2, 4],
        num_res_blocks: int = 2,
        attn_resolutions: List[int] = [3],
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.z_channels = z_channels
        self.double_z = double_z
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temporal_downsample_factor = temporal_downsample_factor
        self.ch = ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout


class Emu3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Emu3Model`]. It is used to instantiate an Emu3
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Emu3-8B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 184622):
            Vocabulary size of the Emu3 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Emu3Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 9216):
            The maximum sequence length that this model might ever be used with. Emu supports up to 9216 tokens,
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, 151643):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 151849):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 151850):
            End of stream token id.
        img_token_id (`int`, *optional*, defaults to 151851):
            image token id.
        boi_token_id (`int`, *optional*, defaults to 151852):
            Beginning of image token id.
        eoi_token_id (`int`, *optional*, defaults to 151853):
            End of image token id.
        eol_token_id (`int`, *optional*, defaults to 151846):
            End of line token id.
        eof_token_id (`int`, *optional*, defaults to 151847):
            End of line token id.
        image_area (`int`, *optional*, defaults to 720 * 720)
            generated image area (image area used in training)
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 1_000_000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import Emu3Model, Emu3Config

    >>> # Initializing a Emu3-8b style configuration
    >>> configuration = Emu3Config()

    >>> # Initializing a model from the Emu3-8b style configuration
    >>> model = Emu3Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "Emu3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 184622,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = 8,
        hidden_act: str = "silu",
        max_position_embeddings: int = 9216,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int = 151643,
        bos_token_id: int = 151849,
        eos_token_id: int = 151850,
        img_token_id: int = 151851,
        boi_token_id: int = 151852,
        eoi_token_id: int = 151853,
        eol_token_id: int = 151846,
        eof_token_id: int = 151847,
        image_area: int = 720 * 720,
        pretraining_tp: int = 1,
        tie_word_embeddings: bool = False,
        rope_theta: float = 1000000.0,
        rope_scaling: Optional[str] = None,
        attention_dropout: float = 0.1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_dropout = attention_dropout

        self.img_token_id = img_token_id
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.eol_token_id = eol_token_id
        self.eof_token_id = eof_token_id
        self.image_area = image_area

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")