# Doctor-S

### Model Introduction

**Doctor Sun**: a bilingual (Chinese-English) MLLM specifically designed to advance medical diagnostics across multiple specialties. Doctor Sun integrates three key components: a text foundation model for logical reasoning and clinical decision-making, a visual foundation model to extract image features and identify abnormalities in medical scans, and a cross-modal projector to align and map visual data into the textual semantic space. This architecture enables the seamless integration of imaging findings with clinical notes, providing a comprehensive understanding of patient conditions. The model is trained on a meticulously curated, high-quality bilingual dataset derived from public sources, encompassing radiology images, pathology slides, and clinical photographs, along with corresponding textual annotations in both Chinese and English. To ensure domain-specific expertise, the general-purpose language foundation model of Doctor Sun is first pre-trained and optimized to accumulate fundamental medical knowledge. Subsequently, the entire model undergoes a two-stage training strategy, focusing on feature alignment and instruction tuning, to achieve proficiency in multimodal medical diagnostic tasks while retaining general-purpose capabilities. 

At present, **Sunsimiao** is fine-tuned from **CLIP** and **LLaMA** of 1, 000, 000 high-quality bilingual multi-modal medical data, and more data will be collected to expand the model's capabilities and iterate on the update. The details are being worked on, so stay tuned.

When the paper is under review, we will release the relevant data, code, and model.


### List of models

| Model Name | weights | 
| :----: | :----: | 
| pretrain | [modelscope] / huggingface | 
| finetune | [modelscope] / huggingface | 


### List of datasets

| dataset | link | 
| :----: | :----: | 
| pretrain | [modelscope] / huggingface | 
| finetune | [modelscope] / huggingface | 

### How to use

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
