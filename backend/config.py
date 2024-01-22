import os
import torch
from fsplit.filesplit import Filesplit

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TRANSLATOR_MODEL_BASE = os.path.join(BASE_DIR, 'models', 'qwen_1_8B')
TRANSLATOR_LANGUAGE = 'English'
TRANSLATOR_PROMPT = os.path.join(BASE_DIR, 'prompt', 'any_to_en_prompt')

if 'model-00001-of-00002.safetensors' not in os.listdir(TRANSLATOR_MODEL_BASE):
    fs = Filesplit()
    fs.merge(input_dir=TRANSLATOR_MODEL_BASE, manifest_file=os.path.join(TRANSLATOR_MODEL_BASE, 'fs_manifest_1.csv'))

if 'model-00002-of-00002.safetensors' not in os.listdir(TRANSLATOR_MODEL_BASE):
    fs = Filesplit()
    fs.merge(input_dir=TRANSLATOR_MODEL_BASE, manifest_file=os.path.join(TRANSLATOR_MODEL_BASE, 'fs_manifest_2.csv'))

