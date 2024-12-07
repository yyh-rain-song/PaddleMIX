# -*- coding: utf-8 -*-

import os
import os.path as osp

from PIL import Image
import paddle
from paddlemix.models.emu3.tokenizer.image_processing_emu3visionvq import Emu3VisionVQImageProcessor
from paddlemix.models.emu3.tokenizer import Emu3VisionVQModel

MODEL_HUB = "BAAI/Emu3-VisionTokenizer"

model = Emu3VisionVQModel.from_pretrained(MODEL_HUB).eval()
processor = Emu3VisionVQImageProcessor.from_pretrained(MODEL_HUB)

# TODO: you need to modify the path here
VIDEO_FRAMES_PATH = "YOUR_VIDEO_FRAMES_PATH"

video = os.listdir(VIDEO_FRAMES_PATH)
video.sort()
video = [Image.open(osp.join(VIDEO_FRAMES_PATH, v)) for v in video]

images = processor(video, return_tensors="pd")["pixel_values"]
images = images.unsqueeze(0)

# image autoencode
image = images[:, 0]
print(image.shape)
with paddle.no_grad():
    # encode
    codes = model.encode(image)
    # decode
    recon = model.decode(codes)

recon = recon.reshape([-1, *recon.shape[2:]])
recon_image = processor.postprocess(recon)["pixel_values"][0]
recon_image.save("recon_image.png")

# video autoencode
images = images.reshape(
    [-1,
    model.config.temporal_downsample_factor,
    *images.shape[2:]],
)

print(images.shape)
with paddle.no_grad():
    # encode
    codes = model.encode(images)
    # decode
    recon = model.decode(codes)

recon = recon.reshape([-1, *recon.shape[2:]])
recon_images = processor.postprocess(recon)["pixel_values"]
for idx, im in enumerate(recon_images):
    im.save(f"recon_video_{idx}.png")
