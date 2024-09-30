# Ensemble of Intermediate-Level Attcks to Boost Adversarial Transferability

Pytorch implement of our paper [Ensemble of Intermediate-Level Attcks to Boost Adversarial Transferability] accepted by ICONIP2024.

## Requirements

```
numpy==1.22.3
PyYAML==6.0.1
scipy==1.11.1
timm==0.6.12
torch==2.0.1
torchvision==0.15.2
tqdm==4.65.0
```
## Dataset

Our evaluation has been done on the ImageNet Dataset. (Implementation on other Datasets will come soon)

First, clone the repository:

` https://github.com/YunCe-Code/EILA.git`

Then the Ensemble of source models used to boost adversarial transferability is included checkpoint.yaml.
If you want to modify the source models utilized to generate adversarial examples, feel free to modify configs/checkpoint.yaml.
And remember to modify the corresponding Intermediate-layers used in utils/eila.py 

Run command python3 main.py 
