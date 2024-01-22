import os
from pathlib import Path
import pysrt
import config
from transformers import AutoModelForCausalLM, AutoTokenizer


class SubtitleTranslator:
    def __init__(self, srt_path, language=config.TRANSLATOR_LANGUAGE):
        self.tokenizer = AutoTokenizer.from_pretrained(config.TRANSLATOR_MODEL_BASE, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(config.TRANSLATOR_MODEL_BASE, device_map="auto", trust_remote_code=True).eval()
        self.language = language
        self.system = self.init_prompt()
        self.srt_path = srt_path
        self.srt_out_path = os.path.join(os.path.dirname(self.srt_path), f'{Path(self.srt_path).stem}_{config.TRANSLATOR_LANGUAGE}.srt')

    def init_prompt(self):
        with open(config.TRANSLATOR_PROMPT, 'r') as f:
            system = f.read()
        return system

    def translate(self, text):
        response, _ = self.model.chat(self.tokenizer, text, history=None, system=self.system)
        return response

    def translate_script(self):
        subs = pysrt.open(self.srt_path, encoding='utf-8')
        with open(self.srt_out_path, 'w') as f:
            for sub in subs:
                f.write(f"{sub.index}\n")
                f.write(f"{sub.start} --> {sub.end}\n")
                f.write(f"{self.translate(sub.text)}\n\n")
                print(f"{sub.index}")
                print(f"{sub.start} --> {sub.end}")
                print(f"{self.translate(sub.text)}\n")  # 打印字幕的文本内容


if __name__ == '__main__':
    srt_path = '/home/yao/Documents/Project/video-subtitle-translator/test/text1.srt'
    st = SubtitleTranslator(srt_path)
    st.translate_script()
