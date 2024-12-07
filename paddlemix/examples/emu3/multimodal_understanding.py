import paddle
from PIL import Image
from paddlemix.models.emu3.mllm.modeling_emu3 import Emu3ForCausalLM
from paddlemix.models.emu3.mllm.tokenization_emu3 import Emu3Tokenizer
from paddlemix.models.emu3.tokenizer import Emu3VisionVQModel
from paddlemix.models.emu3.mllm.processing_emu3 import Emu3Processor
from paddlemix.models.emu3.tokenizer.image_processing_emu3visionvq import Emu3VisionVQImageProcessor
from paddlenlp.generation import GenerationConfig

EMU_HUB = "BAAI/Emu3-Chat"
VQ_HUB = "BAAI/Emu3-VisionTokenizer"

# prepare model and processor
model = Emu3ForCausalLM.from_pretrained(EMU_HUB, dtype=paddle.bfloat16).eval()
tokenizer = Emu3Tokenizer.from_pretrained(EMU_HUB, padding_side="left")

image_processor = Emu3VisionVQImageProcessor.from_pretrained(VQ_HUB)
image_tokenizer = Emu3VisionVQModel.from_pretrained(VQ_HUB).eval()

processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

# prepare input
text = ["Please describe the image"]
image = Image.open("paddlemix/demo_images/emu3_demo.png")
image = [image]

inputs = processor(
    text=text,
    image=image,
    mode='U',
    padding_image=True,
    padding="longest",
    return_tensors="pd",
)
# [1, 4188]
# PretrainedTokenizer(name_or_path='', vocab_size=184622, 
# model_max_len=1000000000000000019884624838656, padding_side='left', truncation_side='right', 
# special_tokens={'bos_token': '<|extra_203|>', 'eos_token': '<|extra_204|>', 'pad_token': '<|endoftext|>'})

# prepare hyper parameters
GENERATION_CONFIG = GenerationConfig(pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)

# generate
outputs = model.generate(
    inputs.input_ids,
    GENERATION_CONFIG,
    max_new_tokens=1024,
    #attention_mask=inputs.attention_mask,
)
#print('outputs', outputs)
answers = processor.batch_decode(outputs[0], skip_special_tokens=True)
for ans in answers:
    print(ans)
