# SAM-finetune
Finetune Meta's SegmentAnything 

This repository contains a wrapper for fine-tuning Meta's SegmentAnything model on a custom dataset
 for a single class. SegmentAnything is a novel deep learning model that performs semantic segmentation on an image for any number of classes. This repo leverages the prompt encoding feature of SegmentAnything to finetune to one class, using bounding boxes around the ground truth masks as the prompting.

Currently supports ViT-H SAM (default) only, tested in KITTI road segmentation data


## Setup

```
conda create --name sam_finetune --file requirements.txt
```

Install [pytorch](https://pytorch.org/) with your version of CUDA, for instance:

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install SegmentAnything

```
pip install git+https://github.com/facebookresearch/segment-anything.git
```


## Data

Run

```
mkdir data
```

and put your dataset according to the following format

```
├── main.py
├── src/
├── data/    
│   ├── your_dataset_name           
│   │   ├── training    
│   │   │   ├── image/          # training RGB images
│   │   │   ├── gt_image/       # training ground truth masks
│   │   ├── testing      
│   │   │   ├── image/          # testing RGB images
│   │   │   ├── gt_image/       # testing ground truth masks (optional)      

```

## Training

```
python main.py --mode train --exp_name your_exp_name --base_dir your_dataset_name
```

Configurable arguments:
* --ckpt_every      : how often (epochs) to save checkpoint
* --save_every      : how often (dataset length) to save training results
* --lr              : learning rate
* --epochs          : num epochs


## Testing

** For best results, run the following to compute an average bbox from trainset to guide test results: ** 

```
python main.py --mode bbox --exp_name your_exp_name --base_dir your_dataset_name
```

**Evaluation**
```
python main.py --mode test --exp_name your_exp_name --base_dir your_dataset_name
```

Configurable argument:
* --ckpt_name       : checkpoint to load, if not provided loads ```best_model.pth```