# Show-o

## 1. 模型介绍

[Show-o](https://github.com/showlab/Show-o) 


## 2 环境准备
- **python >= 3.10**
- **paddlepaddle-gpu 要求版本develop**
```
# 安装示例
python -m pip install paddlepaddle-gpu==0.0.0.post118 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```

- **paddlenlp == 3.0.0b2**

> 注：
* 请确保安装了以上依赖，否则无法运行。同时，需要安装 paddlemix/external_ops 下的自定义OP, `python setup.py install`。如果安装后仍然找不到算子，需要额外设置PYTHONPATH
* (默认开启flash_attn)使用flash_attn 要求A100/A800显卡或者H20显卡

## 3 推理预测

### a. 多模态理解预测
```bash
python paddlemix/examples/showo/inference_mmu.py
```

### b. 多模态生成预测
```bash
python paddlemix/examples/showo/inference_t2i.py
```

## 参考文献
```BibTeX
@article{xie2024showo,
  title={Show-o: One Single Transformer to Unify Multimodal Understanding and Generation},
  author={Xie, Jinheng and Mao, Weijia and Bai, Zechen and Zhang, David Junhao and Wang, Weihao and Lin, Kevin Qinghong and Gu, Yuchao and Chen, Zhijie and Yang, Zhenheng and Shou, Mike Zheng},
  journal={arXiv preprint arXiv:2408.12528},
  year={2024}
}
```
