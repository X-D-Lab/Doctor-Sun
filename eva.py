import argparse
from io import BytesIO

import requests
import torch
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import Conversation, SeparatorStyle
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model import LlavaLlamaForCausalLM
from PIL import Image
from transformers import (AutoTokenizer, BitsAndBytesConfig, StoppingCriteria,
                          StoppingCriteriaList, TextStreamer)

from multimedeval import MultiMedEval, EvalParams, SetupParams
def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


class StopWordStoppingCriteria(StoppingCriteria):
    """StopWord stopping criteria."""

    def __init__(self, tokenizer, stop_word):
        self.tokenizer = tokenizer
        self.stop_word = stop_word
        self.length = len(self.stop_word)

    def __call__(self, input_ids, *args, **kwargs) -> bool:
        cur_text = self.tokenizer.decode(input_ids[0])
        cur_text = cur_text.replace('\r', '').replace('\n', '')
        return cur_text[-self.length:] == self.stop_word


def get_stop_criteria(tokenizer, stop_words=[]):
    stop_criteria = StoppingCriteriaList()
    for word in stop_words:
        stop_criteria.append(StopWordStoppingCriteria(tokenizer, word))
    return stop_criteria

class batcherLLaVA_Med:
    def __init__(self, cacheLocation, args):


        kwargs = {'device_map': args.device}
        if args.load_8bit:
            kwargs['load_in_8bit'] = True
        elif args.load_4bit:
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4')
        else:
            kwargs['torch_dtype'] = torch.float16
    
    
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        self.model = LlavaLlamaForCausalLM.from_pretrained(
            args.model_path, low_cpu_mem_usage=True, **kwargs)
        self.vision_tower = self.model.get_vision_tower()
        if not self.vision_tower.is_loaded:
            self.vision_tower.load_model(device_map=args.device)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.image_processor = self.vision_tower.image_processor
        

    def __call__(self, prompts):

        outputList = []
        listText = []
        images = []
        image_sizes = []
        conv = Conversation(
                system='<|start_header_id|>system<|end_header_id|>\n\nAnswer the questions.',
                roles=('<|start_header_id|>user<|end_header_id|>\n\n',
                        '<|start_header_id|>assistant<|end_header_id|>\n\n'),
                messages=[],
                offset=0,
                sep_style=SeparatorStyle.MPT,
                sep='<|eot_id|>',
            )
        roles = conv.roles
        stop_criteria = get_stop_criteria(
            tokenizer=self.tokenizer, stop_words=[conv.sep])
        streamer = TextStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        img = Image.open('/data/X-D-Lab/med-llava/VQA_RAD/synpic21776.jpg').convert('RGB')
        image_size = img.size
        image_sizes.append(image_size)
        image_tensor = process_images([img], self.image_processor, self.model.config)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        image_tensor = torch.zeros_like(image_tensor) 
        
        images.append(image_tensor)

        
        for prompt in prompts:
            conv = Conversation(
                system='<|start_header_id|>system<|end_header_id|>\n\nAnswer the questions.',
                roles=('<|start_header_id|>user<|end_header_id|>\n\n',
                        '<|start_header_id|>assistant<|end_header_id|>\n\n'),
                messages=[],
                offset=0,
                sep_style=SeparatorStyle.MPT,
                sep='<|eot_id|>',
            )
            roles = conv.roles

            for img in prompt[1]:
                break
                img = img.convert("RGB")
                
                image_size = img.size
                image_sizes.append(image_size)
                image_tensor = process_images([img], self.image_processor, self.model.config)
                image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
                print(image_tensor)
                print(len(image_tensor))
                images.append(image_tensor)


            for message in prompt[0]:
                qs: str = message["content"]
                qs = "What is the second highest mountain in the world?"
                
                qs = qs.replace("<img>", "")
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

                conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            textPrompt = conv.get_prompt()
            

            input_ids = tokenizer_image_token(
                textPrompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                return_tensors='pt').unsqueeze(0).to(self.model.device)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=images,
                    attention_mask=torch.ones_like(input_ids).bool(),
                    image_sizes=image_sizes,
                    do_sample=True ,
                    temperature=0.1,
                    max_new_tokens=512,
                    # streamer=streamer,
                    stopping_criteria=stop_criteria,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id)
            

            outputs = self.tokenizer.decode(output_ids[0]).strip()
            outputs = outputs.replace("<|begin_of_text|>","")
            outputs = outputs.replace("<|eot_id|>","")
            outputs = outputs.replace(":","")

            print(outputs)

            

            outputList.append(outputs.strip())

        return outputList
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-path', type=str, default='/data/X-D-Lab/med-llava/llama3/xtuner/xtuner/configs/llava/llama3_8b_instruct_clip_vit_large_p14_336/finetune/llama3.1-8b-instruct-dpo-zh-/1/iter_23225_llava')
    parser.add_argument('--image-file', type=str, default='/data/X-D-Lab/med-llava/pmcoa/images/images/PMC2180173_F2_15877.jpg')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--temperature', type=float, default=0.95)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--load-8bit', type=bool, default=False)
    parser.add_argument('--load-4bit', type=bool, default=False)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    batcher = batcherLLaVA_Med(
        cacheLocation="/tmp/code/multimedeval/model/",args=args
    )
    print("################## Model has Init! ######################")

    engine = MultiMedEval()
    setupParams = SetupParams(
                         MedQA_dir="/data/X-D-Lab/med-llava/LLaVA-Med-main/multimedeval-dataset"
                         ,PubMedQA_dir="/data/X-D-Lab/med-llava/LLaVA-Med-main/multimedeval-dataset"
                    #    ,MedMCQA_dir="/data/X-D-Lab/med-llava/LLaVA-Med-main/multimedeval-dataset"
                    #   ,MNIST_Path_dir="/data/X-D-Lab/med-llava/LLaVA-Med-main/multimedeval-dataset"
                              ,device="cuda:0")
    engine.setup(setupParams)
    print("$$$$$$$$$$$$$$$ Multimedeval has Init $$$$$$$$$$$$$$$$$$")

    engine.eval(["PubMedQA","MedQA"], batcher, EvalParams(batch_size=1))    
            # engine.eval(["VQA-Rad","VQA-Path","SLAKE","VQA-Path","PubMedQA","MedQA","MedMCQA","OCTMNIST","PathMNIST"], batcher, EvalParams(batch_size=32, run_name="testLLaVAMed"))        
    