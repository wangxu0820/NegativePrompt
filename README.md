## Introduction
The official GitHub page for paper "NegativePrompt: Leveraging Psychology for Large Language Models Enhancement via Negative Emotional Stimuli".

## Installation

First, clone the repo:
```sh
git clone git@https://github.com/wangxu0820/NegativePrompt
```

Then, 

```sh
cd NegativePrompt
```

To install the required packages, you can create a conda environment:

```sh
conda create --name negativeprompt python=3.9
```

then use pip to install required packages:

```sh
pip install -r requirements.txt
```

## Usage
```sh
python main.py --task task_name --model model_name --pnum negativeprompt_id --few_shot False
```

