import os
import torch

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TRANSLATOR_MODEL_BASE = os.path.join(BASE_DIR, 'models', 'qwen_1_8B')
TRANSLATOR_LANGUAGE = 'English'
TRANSLATOR_PROMPT = os.path.join(BASE_DIR, 'prompt', 'any_to_en_prompt')
