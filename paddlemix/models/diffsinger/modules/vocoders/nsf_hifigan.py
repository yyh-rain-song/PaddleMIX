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

import pathlib
import sys
import paddle

from paddlemix.models.diffsinger.utils import paddle_aux
from paddlemix.models.diffsinger.basics.base_vocoder import BaseVocoder
from paddlemix.models.diffsinger.modules.nsf_hifigan.models import load_model
from paddlemix.models.diffsinger.modules.vocoders.registry import register_vocoder
from paddlemix.models.diffsinger.utils.hparams import hparams


@register_vocoder
class NsfHifiGAN(BaseVocoder):
    def __init__(self):
        model_path = pathlib.Path(hparams["vocoder_ckpt"])
        if not model_path.exists():
            raise FileNotFoundError(
                f"NSF-HiFiGAN vocoder model is not found at '{model_path}'. Please follow instructions in docs/BestPractices.md#vocoders to get one."
            )
        print(f"| Load HifiGAN: {model_path}")
        self.model, self.h = load_model(model_path)

    @property
    def device(self):
        # return next(self.model.parameters()).place
        return next(iter(self.model.parameters())).place

    def to_device(self, device):
        self.model.to(device)

    def get_device(self):
        return self.device

    def spec2wav_torch(self, mel, **kwargs):
        if self.h.sampling_rate != hparams["audio_sample_rate"]:
            print(
                "Mismatch parameters: hparams['audio_sample_rate']=",
                hparams["audio_sample_rate"],
                "!=",
                self.h.sampling_rate,
                "(vocoder)",
            )
        if self.h.num_mels != hparams["audio_num_mel_bins"]:
            print(
                "Mismatch parameters: hparams['audio_num_mel_bins']=",
                hparams["audio_num_mel_bins"],
                "!=",
                self.h.num_mels,
                "(vocoder)",
            )
        if self.h.n_fft != hparams["fft_size"]:
            print("Mismatch parameters: hparams['fft_size']=", hparams["fft_size"], "!=", self.h.n_fft, "(vocoder)")
        if self.h.win_size != hparams["win_size"]:
            print("Mismatch parameters: hparams['win_size']=", hparams["win_size"], "!=", self.h.win_size, "(vocoder)")
        if self.h.hop_size != hparams["hop_size"]:
            print("Mismatch parameters: hparams['hop_size']=", hparams["hop_size"], "!=", self.h.hop_size, "(vocoder)")
        if self.h.fmin != hparams["fmin"]:
            print("Mismatch parameters: hparams['fmin']=", hparams["fmin"], "!=", self.h.fmin, "(vocoder)")
        if self.h.fmax != hparams["fmax"]:
            print("Mismatch parameters: hparams['fmax']=", hparams["fmax"], "!=", self.h.fmax, "(vocoder)")
        with paddle.no_grad():
            c = mel.transpose(perm=paddle_aux.transpose_aux_func(mel.ndim, 2, 1))
            mel_base = hparams.get("mel_base", 10)
            if mel_base != "e":
                assert mel_base in [10, "10"], "mel_base must be 'e', '10' or 10."
                c = 2.30259 * c
            f0 = kwargs.get("f0")
            if f0 is not None:
                y = self.model(c, f0).view(-1)
            else:
                y = self.model(c).view(-1)
        return y

    def spec2wav(self, mel, **kwargs):
        if self.h.sampling_rate != hparams["audio_sample_rate"]:
            print(
                "Mismatch parameters: hparams['audio_sample_rate']=",
                hparams["audio_sample_rate"],
                "!=",
                self.h.sampling_rate,
                "(vocoder)",
            )
        if self.h.num_mels != hparams["audio_num_mel_bins"]:
            print(
                "Mismatch parameters: hparams['audio_num_mel_bins']=",
                hparams["audio_num_mel_bins"],
                "!=",
                self.h.num_mels,
                "(vocoder)",
            )
        if self.h.n_fft != hparams["fft_size"]:
            print("Mismatch parameters: hparams['fft_size']=", hparams["fft_size"], "!=", self.h.n_fft, "(vocoder)")
        if self.h.win_size != hparams["win_size"]:
            print("Mismatch parameters: hparams['win_size']=", hparams["win_size"], "!=", self.h.win_size, "(vocoder)")
        if self.h.hop_size != hparams["hop_size"]:
            print("Mismatch parameters: hparams['hop_size']=", hparams["hop_size"], "!=", self.h.hop_size, "(vocoder)")
        if self.h.fmin != hparams["fmin"]:
            print("Mismatch parameters: hparams['fmin']=", hparams["fmin"], "!=", self.h.fmin, "(vocoder)")
        if self.h.fmax != hparams["fmax"]:
            print("Mismatch parameters: hparams['fmax']=", hparams["fmax"], "!=", self.h.fmax, "(vocoder)")
        with paddle.no_grad():
            c = (
                paddle.to_tensor(data=mel, dtype="float32")
                .unsqueeze(axis=0)
                .transpose(
                    perm=paddle_aux.transpose_aux_func(
                        paddle.to_tensor(data=mel, dtype="float32").unsqueeze(axis=0).ndim, 2, 1
                    )
                )
                .to(self.device)
            )
            mel_base = hparams.get("mel_base", 10)
            if mel_base != "e":
                assert mel_base in [10, "10"], "mel_base must be 'e', '10' or 10."
                c = 2.30259 * c
            f0 = kwargs.get("f0")
            if f0 is not None:
                f0 = paddle.to_tensor(data=f0[None, :], dtype="float32").to(self.device)
                y = self.model(c, f0).view(-1)
            else:
                y = self.model(c).view(-1)
        wav_out = y.cpu().numpy()
        return wav_out
