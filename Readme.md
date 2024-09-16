![DALL-e](https://github.com/user-attachments/assets/8ff86ca8-de4a-415d-9e09-4cb341316d40)<h2 align="center">PoliStance-TR: A Dataset for Turkish Stance Detection in Political Domain</h2>
<p align="center">
  Developed by <a href="https://github.com/ByUnal"> M.Cihat Unal </a> 
</p>

This repository introduces a new dataset for **Turkish Stance Detection** and provides code for fine-tuning transformer-based models on this dataset. The dataset includes three stance labels: **Favor**, **Against**, and **Neutral**.

## Dataset Overview

The dataset was specifically collected for stance detection in the Turkish language. It contains the following labels:
- **Favor**: The text supports the target.
- **Against**: The text opposes the target.
- **Neutral**: The text does not express a clear stance on the target.

### Data Splits and Distribution

The dataset is split into three parts as follows:
- **Train data**: 6060 samples
- **Validation data**: 674 samples
- **Test data**: 1189 samples

Each set retains the same percentage of labels as the original dataset. The overall label distribution is:
- **Favor (Positive)**: 2898 samples
- **Against (Negative)**: 2858 samples
- **Neutral**: 2167 samples

The data files are located in the `data/` folder:
- `stance_train.csv`s
- `stance_val.csv`
- `stance_test.csv`

## Model Fine-Tuning

We provide `main.py` as the primary script for fine-tuning pre-trained transformer-based models on this dataset. The models have been trained to classify stance into the three categories (Favor, Against, Neutral) using this unique Turkish stance detection dataset.

### Key Files:
- `main.py`: Main script for model fine-tuning on Turkish stance detection.
- `preprocess.py`: Includes necessary scripts for preprocessing.
- Jupyten Notebook file which includes all necessary codes for both training and evaluation can be found in `notebook/` folder.


## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/ByUnal/polistance-tr.git
   cd polistance-tr

2. Install Dependencies:
    ```bash
   pip install -r requirements.txt
   
3. Fine-tune the model
    ```bash
   python main.py --learning_rate 4e-5 --epoch 10 --save_dir trained-models

if you want to push the model after fine-tuning to HuggingFace enter the repository name by using ``--hf_repo_name`` 
environment variable.

## Pre-trained Models
Transformer-based Fine-tuned models can be reached via my [HuggingFace profile](https://huggingface.co/byunal).
