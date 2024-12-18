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

from __future__ import annotations

import sys
from typing import List, Tuple
import paddle

from tqdm import tqdm
from paddlemix.models.diffsinger.modules.backbones import build_backbone
from paddlemix.models.diffsinger.utils.hparams import hparams
from paddlemix.models.diffsinger.utils import paddle_aux

class RectifiedFlow(paddle.nn.Layer):
    def __init__(
        self,
        out_dims,
        num_feats=1,
        t_start=0.0,
        time_scale_factor=1000,
        backbone_type=None,
        backbone_args=None,
        spec_min=None,
        spec_max=None,
    ):
        super().__init__()
        self.velocity_fn: paddle.nn.Layer = build_backbone(out_dims, num_feats, backbone_type, backbone_args)
        self.out_dims = out_dims
        self.num_feats = num_feats
        self.use_shallow_diffusion = hparams.get("use_shallow_diffusion", False)
        if self.use_shallow_diffusion:
            assert 0.0 <= t_start <= 1.0, "T_start should be in [0, 1]."
        else:
            t_start = 0.0
        self.t_start = t_start
        self.time_scale_factor = time_scale_factor
        spec_min = paddle.to_tensor(data=spec_min, dtype="float32")[None, None, :out_dims].transpose(
            perm=paddle_aux.transpose_aux_func(
                paddle.to_tensor(data=spec_min, dtype="float32")[None, None, :out_dims].ndim, -3, -2
            )
        )
        spec_max = paddle.to_tensor(data=spec_max, dtype="float32")[None, None, :out_dims].transpose(
            perm=paddle_aux.transpose_aux_func(
                paddle.to_tensor(data=spec_max, dtype="float32")[None, None, :out_dims].ndim, -3, -2
            )
        )
        self.register_buffer(name="spec_min", tensor=spec_min, persistable=False)
        self.register_buffer(name="spec_max", tensor=spec_max, persistable=False)

    def p_losses(self, x_end, t, cond):
        x_start = paddle.randn(shape=x_end.shape, dtype=x_end.dtype)
        x_t = x_start + t[:, None, None, None] * (x_end - x_start)
        v_pred = self.velocity_fn(x_t, t * self.time_scale_factor, cond)
        return v_pred, x_end - x_start

    def forward(self, condition, gt_spec=None, src_spec=None, infer=True):
        cond = condition.transpose(perm=paddle_aux.transpose_aux_func(condition.ndim, 1, 2))
        b, device = tuple(condition.shape)[0], condition.place
        if not infer:
            spec = self.norm_spec(gt_spec).transpose(
                perm=paddle_aux.transpose_aux_func(self.norm_spec(gt_spec).ndim, -2, -1)
            )
            if self.num_feats == 1:
                spec = spec[:, None, :, :]
            t = self.t_start + (1.0 - self.t_start) * paddle.rand(shape=(b,))
            v_pred, v_gt = self.p_losses(spec, t, cond=cond)
            return v_pred, v_gt, t
        else:
            if src_spec is not None:
                spec = self.norm_spec(src_spec).transpose(
                    perm=paddle_aux.transpose_aux_func(self.norm_spec(src_spec).ndim, -2, -1)
                )
                if self.num_feats == 1:
                    spec = spec[:, None, :, :]
            else:
                spec = None
            x = self.inference(cond, b=b, x_end=spec, device=device)
            return self.denorm_spec(x)

    @paddle.no_grad()
    def sample_euler(self, x, t, dt, cond):
        x += self.velocity_fn(x, self.time_scale_factor * t, cond) * dt
        t += dt
        return x, t

    @paddle.no_grad()
    def sample_rk2(self, x, t, dt, cond):
        k_1 = self.velocity_fn(x, self.time_scale_factor * t, cond)
        k_2 = self.velocity_fn(x + 0.5 * k_1 * dt, self.time_scale_factor * (t + 0.5 * dt), cond)
        x += k_2 * dt
        t += dt
        return x, t

    @paddle.no_grad()
    def sample_rk4(self, x, t, dt, cond):
        k_1 = self.velocity_fn(x, self.time_scale_factor * t, cond)
        k_2 = self.velocity_fn(x + 0.5 * k_1 * dt, self.time_scale_factor * (t + 0.5 * dt), cond)
        k_3 = self.velocity_fn(x + 0.5 * k_2 * dt, self.time_scale_factor * (t + 0.5 * dt), cond)
        k_4 = self.velocity_fn(x + k_3 * dt, self.time_scale_factor * (t + dt), cond)
        x += (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt / 6
        t += dt
        return x, t

    @paddle.no_grad()
    def sample_rk5(self, x, t, dt, cond):
        k_1 = self.velocity_fn(x, self.time_scale_factor * t, cond)
        k_2 = self.velocity_fn(x + 0.25 * k_1 * dt, self.time_scale_factor * (t + 0.25 * dt), cond)
        k_3 = self.velocity_fn(x + 0.125 * (k_2 + k_1) * dt, self.time_scale_factor * (t + 0.25 * dt), cond)
        k_4 = self.velocity_fn(x + 0.5 * (-k_2 + 2 * k_3) * dt, self.time_scale_factor * (t + 0.5 * dt), cond)
        k_5 = self.velocity_fn(x + 0.0625 * (3 * k_1 + 9 * k_4) * dt, self.time_scale_factor * (t + 0.75 * dt), cond)
        k_6 = self.velocity_fn(
            x + (-3 * k_1 + 2 * k_2 + 12 * k_3 - 12 * k_4 + 8 * k_5) * dt / 7, self.time_scale_factor * (t + dt), cond
        )
        x += (7 * k_1 + 32 * k_3 + 12 * k_4 + 32 * k_5 + 7 * k_6) * dt / 90
        t += dt
        return x, t

    @paddle.no_grad()
    def inference(self, cond, b=1, x_end=None, device=None):
        noise = paddle.randn(shape=[b, self.num_feats, self.out_dims, tuple(cond.shape)[2]])
        t_start = hparams.get("T_start_infer", self.t_start)
        if self.use_shallow_diffusion and t_start > 0:
            assert x_end is not None, "Missing shallow diffusion source."
            if t_start >= 1.0:
                t_start = 1.0
                x = x_end
            else:
                x = t_start * x_end + (1 - t_start) * noise
        else:
            t_start = 0.0
            x = noise
        algorithm = hparams["sampling_algorithm"]
        infer_step = hparams["sampling_steps"]
        if t_start < 1:
            dt = (1.0 - t_start) / max(1, infer_step)
            algorithm_fn = {
                "euler": self.sample_euler,
                "rk2": self.sample_rk2,
                "rk4": self.sample_rk4,
                "rk5": self.sample_rk5,
            }.get(algorithm)
            if algorithm_fn is None:
                raise ValueError(f"Unsupported algorithm for Rectified Flow: {algorithm}.")
            dts = paddle.to_tensor(data=[dt]).to(x)
            for i in tqdm(
                range(infer_step), desc="sample time step", total=infer_step, disable=not hparams["infer"], leave=False
            ):
                x, _ = algorithm_fn(x, t_start + i * dts, dt, cond)
            x = x.astype(dtype="float32")
        x = x.transpose(perm=paddle_aux.transpose_aux_func(x.ndim, 2, 3)).squeeze(axis=1)
        return x

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min


class RepetitiveRectifiedFlow(RectifiedFlow):
    def __init__(
        self,
        vmin: (float | int | list),
        vmax: (float | int | list),
        repeat_bins: int,
        time_scale_factor=1000,
        backbone_type=None,
        backbone_args=None,
    ):
        assert isinstance(vmin, (float, int)) and isinstance(vmin, (float, int)) or len(vmin) == len(vmax)
        num_feats = 1 if isinstance(vmin, (float, int)) else len(vmin)
        spec_min = [vmin] if num_feats == 1 else [[v] for v in vmin]
        spec_max = [vmax] if num_feats == 1 else [[v] for v in vmax]
        self.repeat_bins = repeat_bins
        super().__init__(
            out_dims=repeat_bins,
            num_feats=num_feats,
            time_scale_factor=time_scale_factor,
            backbone_type=backbone_type,
            backbone_args=backbone_args,
            spec_min=spec_min,
            spec_max=spec_max,
        )

    def norm_spec(self, x):
        """

        :param x: [B, T] or [B, F, T]
        :return [B, T, R] or [B, F, T, R]
        """
        if self.num_feats == 1:
            repeats = [1, 1, self.repeat_bins]
        else:
            repeats = [1, 1, 1, self.repeat_bins]
        return super().norm_spec(x.unsqueeze(axis=-1).tile(repeat_times=repeats))

    def denorm_spec(self, x):
        """

        :param x: [B, T, R] or [B, F, T, R]
        :return [B, T] or [B, F, T]
        """
        return super().denorm_spec(x).mean(axis=-1)


class PitchRectifiedFlow(RepetitiveRectifiedFlow):
    def __init__(
        self,
        vmin: float,
        vmax: float,
        cmin: float,
        cmax: float,
        repeat_bins,
        time_scale_factor=1000,
        backbone_type=None,
        backbone_args=None,
    ):
        self.vmin = vmin
        self.vmax = vmax
        self.cmin = cmin
        self.cmax = cmax
        super().__init__(
            vmin=vmin,
            vmax=vmax,
            repeat_bins=repeat_bins,
            time_scale_factor=time_scale_factor,
            backbone_type=backbone_type,
            backbone_args=backbone_args,
        )

    def norm_spec(self, x):
        return super().norm_spec(x.clip(min=self.cmin, max=self.cmax))

    def denorm_spec(self, x):
        return super().denorm_spec(x).clip(min=self.cmin, max=self.cmax)


class MultiVarianceRectifiedFlow(RepetitiveRectifiedFlow):
    def __init__(
        self,
        ranges: List[Tuple[float, float]],
        clamps: List[Tuple[float | None, float | None] | None],
        repeat_bins,
        time_scale_factor=1000,
        backbone_type=None,
        backbone_args=None,
    ):
        assert len(ranges) == len(clamps)
        self.clamps = clamps
        vmin = [r[0] for r in ranges]
        vmax = [r[1] for r in ranges]
        if len(vmin) == 1:
            vmin = vmin[0]
        if len(vmax) == 1:
            vmax = vmax[0]
        super().__init__(
            vmin=vmin,
            vmax=vmax,
            repeat_bins=repeat_bins,
            time_scale_factor=time_scale_factor,
            backbone_type=backbone_type,
            backbone_args=backbone_args,
        )

    def clamp_spec(self, xs: (list | tuple)):
        clamped = []
        for x, c in zip(xs, self.clamps):
            if c is None:
                clamped.append(x)
                continue
            clamped.append(x.clip(min=c[0], max=c[1]))
        return clamped

    def norm_spec(self, xs: (list | tuple)):
        """

        :param xs: sequence of [B, T]
        :return: [B, F, T] => super().norm_spec(xs) => [B, F, T, R]
        """
        assert len(xs) == self.num_feats
        clamped = self.clamp_spec(xs)
        xs = paddle.stack(x=clamped, axis=1)
        if self.num_feats == 1:
            xs = xs.squeeze(axis=1)
        return super().norm_spec(xs)

    def denorm_spec(self, xs):
        """

        :param xs: [B, T, R] or [B, F, T, R] => super().denorm_spec(xs) => [B, T] or [B, F, T]
        :return: sequence of [B, T]
        """
        xs = super().denorm_spec(xs)
        if self.num_feats == 1:
            xs = [xs]
        else:
            xs = xs.unbind(axis=1)
        assert len(xs) == self.num_feats
        return self.clamp_spec(xs)
