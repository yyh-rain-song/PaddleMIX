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

from typing import TYPE_CHECKING, Any, Dict

from .chatter import WebChatModel
from .common import get_time
from .locales import LOCALES
from .manager import Manager
from .runner import Runner

if TYPE_CHECKING:
    from gradio.components import Component


class Engine:
    def __init__(self, demo_mode: bool = False, pure_chat: bool = False) -> None:
        self.demo_mode = demo_mode
        self.pure_chat = pure_chat
        self.manager = Manager()
        self.runner = Runner(self.manager, demo_mode)
        self.chatter = WebChatModel(self.manager, demo_mode, lazy_init=(not pure_chat))

    def _update_component(self, input_dict: Dict[str, Dict[str, Any]]) -> Dict["Component", "Component"]:
        r"""
        Gets the dict to update the components.
        """
        output_dict: Dict["Component", "Component"] = {}
        for elem_id, elem_attr in input_dict.items():
            elem = self.manager.get_elem_by_id(elem_id)
            output_dict[elem] = elem.__class__(**elem_attr)

        return output_dict

    def resume(self):
        init_dict = {
            "top.lang": {"value": "zh"},
            "top.model_name": {"value": "Qwen2-VL-2B-Instruct"},
            "top.model_path": {"value": "Qwen/Qwen2-VL-2B-Instruct"},
        }

        if not self.pure_chat:
            current_time = get_time()
            init_dict["train.current_time"] = {"value": current_time}
            init_dict["train.output_dir"] = {"value": f"train_{current_time}"}
            init_dict["train.config_path"] = {"value": f"{current_time}.yaml"}

        yield self._update_component(init_dict)

        if self.runner.running and not self.pure_chat:
            yield {elem: elem.__class__(value=value) for elem, value in self.runner.running_data.items()}
            if self.runner.do_train:
                yield self._update_component({"train.resume_btn": {"value": True}})
            else:
                yield self._update_component({"eval.resume_btn": {"value": True}})

    def change_lang(self, lang: str):
        return {
            elem: elem.__class__(**LOCALES[elem_name][lang])
            for elem_name, elem in self.manager.get_elem_iter()
            if elem_name in LOCALES
        }
