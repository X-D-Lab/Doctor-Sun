# Doctor Sun: A bilingual multimodal large language model for biomedical AI A bilingual MLLM for biomedical AI

### Model Introduction

**Doctor Sun**: a bilingual (Chinese-English) MLLM specifically designed to advance medical diagnostics across multiple specialties. Doctor Sun integrates three key components: a text foundation model for logical reasoning and clinical decision-making, a visual foundation model to extract image features and identify abnormalities in medical scans, and a cross-modal projector to align and map visual data into the textual semantic space. This architecture enables the seamless integration of imaging findings with clinical notes, providing a comprehensive understanding of patient conditions. The model is trained on a meticulously curated, high-quality bilingual dataset derived from public sources, encompassing radiology images, pathology slides, and clinical photographs, along with corresponding textual annotations in both Chinese and English. To ensure domain-specific expertise, the general-purpose language foundation model of Doctor Sun is first pre-trained and optimized to accumulate fundamental medical knowledge. Subsequently, the entire model undergoes a two-stage training strategy, focusing on feature alignment and instruction tuning, to achieve proficiency in multimodal medical diagnostic tasks while retaining general-purpose capabilities. 

At present, **Doctor Sun** is fine-tuned from **CLIP** and **LLaMA** of high-quality bilingual multi-modal medical data, and more data will be collected to expand the model's capabilities and iterate on the update. The details are being worked on, so stay tuned.


### Overview of the code and dataset


This warehouse contains all the training and evaluation codes for Doctor Sun. 
The "xtuner.zip" file is a training code archive.
The folder “llava” and eva.py contain the evaluation code.


For the dataset, please refer to (https://www.modelscope.cn/datasets/Yanlan/Doctor-Sun-VL). It includes the complete pre-training and fine-tuning datasets.
finetune.json: Fine-tuning text file
image_finetune.zip: Fine-tuned image compression package
pretrain.json: Pre-trained text file 
image_pretrain.zip: Pre-trained Image Compression Package


### Dataset Information
<img width="852" height="762" alt="image" src="https://github.com/user-attachments/assets/081d3689-e4e8-4bbf-b51e-67ddb414b74c" />

In the dataset repository, you can find the following files:
finetune.json: Fine-tuning text file
image_finetune.zip: Fine-tuned image compression package
pretrain.json: Pre-trained text file 
image_pretrain.zip: Pre-trained Image Compression Package

All the datasets can be publicly downloaded.



###  ‎Code Information

The "xtuner.zip" file is a training code archive.
The folder “llava” and eva.py contain the evaluation code.


### List of models

| Model Name | weights | 
| :----: | :----: | 
| pretrain | [modelscope] / huggingface | 
| finetune | [modelscope](https://www.modelscope.cn/models/Yanlan/Doctor-Sun/files) / huggingface | 


### List of datasets

| dataset | link | 
| :----: | :----: | 
| pretrain | [modelscope](https://www.modelscope.cn/datasets/Yanlan/Doctor-Sun-VL) / huggingface | 
| finetune | [modelscope](https://www.modelscope.cn/datasets/Yanlan/Doctor-Sun-VL) / huggingface | 

### How to use


1. Clone this repository and navigate
```bash
git clone https://github.com/X-D-Lab/Doctor-Sun
cd Doctor-Sun
unzip xtuner.zip
```

2. Install Package
```Shell
conda create -n DoctorS python=3.10 -y
conda activate DoctorS
pip install --upgrade pip 
pip install -r requirements.txt
```
3. Quick Start 

# Pretrain

```Shell
NPROC_PER_NODE=4 xtuner train ./xtuner/xtuner/configs/llava/llama3_8b_instruct_clip_vit_large_p14_336/pretrain/llava_llama3_8b_instruct_clip_vit_large_p14_336_e1_gpu8_sharegpt4v_pretrain --deepspeed deepspeed_zero2 --seed 1024
```

# Finetune

```Shell
NPROC_PER_NODE=4 xtuner ./xtuner/xtuner/configs/llava/llama3_8b_instruct_clip_vit_large_p14_336/finetune/llava_llama3_8b_instruct_full_clip_vit_large_p14_336_lora_e1_gpu8_internvl_finetune.py --deepspeed deepspeed_zero2 --seed 1024

cd ./xtuner/xtuner/configs/llava/llama3_8b_instruct_clip_vit_large_p14_336/finetune/work_dirs/llava_llama3_8b_instruct_full_clip_vit_large_p14_336_lora_e1_gpu8_internvl_finetune
num=23226
xtuner convert pth_to_hf llava_llama3_8b_instruct_full_clip_vit_large_p14_336_lora_e1_gpu8_internvl_finetune ./iter_${num}.pth ./iter_${num}_xtuner


xtuner convert merge /home/models/clip-vit-large-patch14-336 ./iter_${num}_xtuner/visual_encoder_adapter ./iter_${num}_visual_encoder --is-clip


python ./xtuner/xtuner/configs/llava/llama3_8b_instruct_clip_vit_large_p14_336/convert_xtuner_weights_to_llava.py --text_model_id ./iter_${num}_xtuner --vision_model_id ./iter_${num}_visual_encoder --projector_weight ./iter_${num}_xtuner/projector/model.safetensors --save_path ./iter_${num}_llava
```
The model "clip-vit-large-patch14-336" needs to be downloaded to a specific location in advance.


4. Evaluation

Refer to the file "eva.py", which is used to evaluate the QA task. With a few simple modifications, it can also be used to evaluate the VQA task.

```Shell
python eva.py
or
HF_ENDPOINT=https://hf-mirror.com python eva.py
```

### 引用

```

@misc{2024Doctor-Sun, 
  author={Dong Xue*, Ziyao Shao, Zhaoyang Duan, Fangzhou Liu, Bing Li, and Zhongheng Zhang*}, 
  title = {Doctor Sun: A Bilingual Multimodal Large Language Model for Biomedical AI}, 
  year = {2024}, 
  publisher = {GitHub}, 
  journal = {GitHub repository}, 
  howpublished = {\url{https://github.com/X-D-Lab/Doctor-Sun/}}, 
}
```
