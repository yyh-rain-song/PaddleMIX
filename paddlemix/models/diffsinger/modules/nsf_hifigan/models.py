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

import json
import pathlib
import sys
import numpy as np
import paddle
import paddle.nn.functional as F

from paddlemix.models.diffsinger.utils import paddle_aux
from paddle.nn.utils import remove_weight_norm, weight_norm

from .env import AttrDict
from .utils import get_padding, init_weights

LRELU_SLOPE = 0.1


def load_model(model_path: pathlib.Path):
    config_file = model_path.with_name("config.json")
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    generator = Generator(h)
    cp_dict = paddle.load(path=str(model_path))
    generator.set_state_dict(state_dict=cp_dict["generator"])
    generator.eval()
    generator.remove_weight_norm()
    del cp_dict
    return generator, h


class ResBlock1(paddle.nn.Layer):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = paddle.nn.LayerList(
            sublayers=[
                paddle.nn.utils.weight_norm(
                    layer=paddle.nn.Conv1D(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                paddle.nn.utils.weight_norm(
                    layer=paddle.nn.Conv1D(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                paddle.nn.utils.weight_norm(
                    layer=paddle.nn.Conv1D(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)
        self.convs2 = paddle.nn.LayerList(
            sublayers=[
                paddle.nn.utils.weight_norm(
                    layer=paddle.nn.Conv1D(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                paddle.nn.utils.weight_norm(
                    layer=paddle.nn.Conv1D(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                paddle.nn.utils.weight_norm(
                    layer=paddle.nn.Conv1D(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = paddle.nn.functional.leaky_relu(x=x, negative_slope=LRELU_SLOPE)
            xt = c1(xt)
            xt = paddle.nn.functional.leaky_relu(x=xt, negative_slope=LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            paddle.nn.utils.remove_weight_norm(layer=l)
        for l in self.convs2:
            paddle.nn.utils.remove_weight_norm(layer=l)


class ResBlock2(paddle.nn.Layer):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = paddle.nn.LayerList(
            sublayers=[
                paddle.nn.utils.weight_norm(
                    layer=paddle.nn.Conv1D(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                paddle.nn.utils.weight_norm(
                    layer=paddle.nn.Conv1D(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = paddle.nn.functional.leaky_relu(x=x, negative_slope=LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            paddle.nn.utils.remove_weight_norm(layer=l)


class SineGen(paddle.nn.Layer):
    """Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-waveform (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_threshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voiced_threshold=0):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        uv = paddle.ones_like(x=f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0, upp):
        """f0: (batchsize, length, dim)
        where dim indicates fundamental tone and overtones
        """
        # rad = f0 / self.sampling_rate * paddle.arange(start=1, end=upp + 1)
        rad = f0 / self.sampling_rate * paddle.arange(start=1, end=upp + 1, dtype="float32")
        rad2 = (
            paddle.mod(
                x=rad[..., -1:].astype(dtype="float32") + 0.5,
                y=paddle.to_tensor(1.0, dtype=(rad[..., -1:].astype(dtype="float32") + 0.5).dtype),
            )
            - 0.5
        )
        rad_acc = rad2.cumsum(axis=1).mod(y=paddle.to_tensor(1.0)).to(f0)
        # rad += paddle.nn.functional.pad(x=rad_acc, pad=(0, 0, 1, -1),
        #     pad_from_left_axis=False)
        # 等效实现
        rad_shifted = paddle.concat([paddle.zeros_like(rad_acc[:, :1]), rad_acc[:, :-1]], axis=1)
        rad += rad_shifted
        rad = rad.reshape(tuple(f0.shape)[0], -1, 1)
        # rad = paddle.multiply(x=rad, y=paddle.to_tensor(paddle.arange(start
        #     =1, end=self.dim + 1).reshape(1, 1, -1)))
        rad = paddle.multiply(
            x=rad,
            y=paddle.to_tensor(
                paddle.arange(start=1, end=self.dim + 1), dtype="float32"  # Explicitly set dtype to float32
            ).reshape(1, 1, -1),
        )

        rand_ini = paddle.rand(shape=[1, 1, self.dim])
        rand_ini[..., 0] = 0
        rad += rand_ini
        sines = paddle.sin(x=2 * np.pi * rad)
        return sines

    @paddle.no_grad()
    def forward(self, f0, upp):
        """sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        f0 = f0.unsqueeze(axis=-1)
        sine_waves = self._f02sine(f0, upp) * self.sine_amp
        uv = (f0 > self.voiced_threshold).astype(dtype="float32")
        uv = F.interpolate(uv.transpose([0, 2, 1]), scale_factor=upp, mode="linear", data_format="NCW").transpose(
            [0, 2, 1]
        )
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * paddle.randn(shape=sine_waves.shape, dtype=sine_waves.dtype)
        sine_waves = sine_waves * uv + noise
        return sine_waves


class SourceModuleHnNSF(paddle.nn.Layer):
    """SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1, add_noise_std=0.003, voiced_threshold=0):
        super(SourceModuleHnNSF, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold)
        self.l_linear = paddle.nn.Linear(in_features=harmonic_num + 1, out_features=1)
        self.l_tanh = paddle.nn.Tanh()

    def forward(self, x, upp):
        sine_wavs = self.l_sin_gen(x, upp)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge


class Generator(paddle.nn.Layer):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.m_source = SourceModuleHnNSF(sampling_rate=h.sampling_rate, harmonic_num=8)
        self.noise_convs = paddle.nn.LayerList()
        self.conv_pre = paddle.nn.utils.weight_norm(
            layer=paddle.nn.Conv1D(
                in_channels=h.num_mels, out_channels=h.upsample_initial_channel, kernel_size=7, stride=1, padding=3
            )
        )
        resblock = ResBlock1 if h.resblock == "1" else ResBlock2
        self.ups = paddle.nn.LayerList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            c_cur = h.upsample_initial_channel // 2 ** (i + 1)
            self.ups.append(
                paddle.nn.utils.weight_norm(
                    layer=paddle.nn.Conv1DTranspose(
                        in_channels=h.upsample_initial_channel // 2**i,
                        out_channels=h.upsample_initial_channel // 2 ** (i + 1),
                        kernel_size=k,
                        stride=u,
                        padding=(k - u) // 2,
                    )
                )
            )
            if i + 1 < len(h.upsample_rates):
                stride_f0 = int(np.prod(h.upsample_rates[i + 1 :]))
                self.noise_convs.append(
                    paddle.nn.Conv1D(
                        in_channels=1,
                        out_channels=c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(paddle.nn.Conv1D(in_channels=1, out_channels=c_cur, kernel_size=1))
        self.resblocks = paddle.nn.LayerList()
        ch = h.upsample_initial_channel
        for i in range(len(self.ups)):
            ch //= 2
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))
        self.conv_post = paddle.nn.utils.weight_norm(
            layer=paddle.nn.Conv1D(in_channels=ch, out_channels=1, kernel_size=7, stride=1, padding=3)
        )
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.upp = int(np.prod(h.upsample_rates))

    def forward(self, x, f0):
        har_source = self.m_source(f0, self.upp).transpose(
            perm=paddle_aux.transpose_aux_func(self.m_source(f0, self.upp).ndim, 1, 2)
        )
        # har_source = self.m_source(f0, self.upp).transpose(1, 2)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = paddle.nn.functional.leaky_relu(x=x, negative_slope=LRELU_SLOPE)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = paddle.nn.functional.leaky_relu(x=x)
        x = self.conv_post(x)
        x = paddle.nn.functional.tanh(x=x)
        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
