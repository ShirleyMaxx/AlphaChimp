<div align="center">

  <h1 align="center">AlphaChimp: Tracking and Behavior Recognition of Chimpanzees</h1>
  
</div>

Offical [Pytorch](https://pytorch.org/) implementation of our paper:
<h3 align="center">AlphaChimp: Tracking and Behavior Recognition of Chimpanzees</h3>

<h4 align="center" style="text-decoration: none;">
  <a href="https://shirleymaxx.github.io/", target="_blank">Xiaoxuan Ma</a><sup>*</sup>
  ,
  <a href="https://github.com/Yutang-Lin", target="_blank">Yutang Lin</a><sup>*</sup>
  ,
  <a href="https://xy02-05.github.io/", target="_blank">Yuan Xu</a>
  ,
  <a href="https://carta.anthropogeny.org/users/stephan-kaufhold", target="_blank">Stephan P. Kaufhold</a><sup></sup>
  ,
  <a href="http://jackterwilliger.com/", target="_blank">Jack Terwilliger</a>
  ,
  <a href="https://www.linkedin.com/in/andy-meza-9bb064213/", target="_blank">Andres Meza</a>
  ,
  <a href="https://yzhu.io/", target="_blank">Yixin Zhu</a>
  ,
  <a href="https://cogsci.ucsd.edu/people/faculty/federico-rossano.html", target="_blank">Federico Rossano</a>
  ,
  <a href="https://cfcs.pku.edu.cn/english/people/faculty/yizhouwang/index.htm", target="_blank">Yizhou Wang</a>
</h4>
<h4 align="center">
  <a href="https://arxiv.org/abs/2410.17136", target="_blank">[Paper]</a> /
  <a href="https://sites.google.com/view/alphachimp/home", target="_blank">[Project page]</a> /
  <a href="https://github.com/ShirleyMaxx/ChimpACT", target="_blank">[ChimpACT dataset]</a>
</h4>

<p align="center">
  <img src="demo/teaser.gif"/>
</p>

# Installation

Clone this project. NVIDIA GPUs are needed. 
```bash
git clone https://github.com/ShirleyMaxx/AlphaChimp
cd AlphaChimp

conda create -n alphachimp python=3.8
conda activate alphachimp

conda install ffmpeg

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -v -e .
    
pip install openmim
mim install mmcv==2.1.0
    
pip install pycocotools shapely terminaltables imageio[ffmpeg] lap
```

# Data

1. Please follow the instructions to **download** and **preprocess** [ChimpACT dataset](https://github.com/ShirleyMaxx/ChimpACT?tab=readme-ov-file#data) and place it under

    ```
    AlphaChimp/
        ...
        data/
            ChimpACT_processed/
                annotations/
                    action/
                    ...
                ...
                test/
                    ...
                train/
                    ...
                ...
        ...
    ```

2. Download our pretrained [checkpoints](https://mega.nz/folder/k2lXxITL#VD5CQv-fBc_9eCRoIaY_-Q) and place them under `work_dirs/alphachimp`. We provide checkpoints with different resolutions.


# Quick Demo :star:
We support inference on videos containing chimpanzees. 

  1. Put the videos under directory `infer_input`. 
  2. Specify the number of GPUs using `${NGPU}`. 
  3. Choose argument `vis_mode` in [`det`, `act`, `mix`] to visualize detection, action or both. 
  4. By default, we save the visualized results in `infer_output`.
  5. Run
      ```
      python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" \
          --nproc_per_node=${NGPU} --master_port=22525 tools/inference.py \
          configs/alphachimp/alphachimp_infer576.py \
          --vis_mode 'mix' \
          --gpus ${NGPU}
      ```

# Train (Action & Detection)
We support different resolutions, choose `${RES}=256 or 576`. Specify the number of GPUs using `${NGPU}`. We support [DDP](https://pytorch.org/tutorials/beginner/ddp_series_theory.html) training. Run
```
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 \
    --nproc_per_node=${NGPU} --master_port=25525 tools/train.py \
    configs/alphachimp/alphachimp_res${RES}.py
```

# Eval (Action & Detection)
Specify resolution `${RES}` and number of GPUs `${NGPU}`. We set default model checkpoint for evaluation. Change it by `--checkpoint`. Run
```
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 \
    --nproc_per_node=${NGPU} --master_port=25525 tools/test.py \
    configs/alphachimp/alphachimp_res${RES}.py \
    --checkpoint work_dirs/alphachimp/alphachimp_res${RES}.pth
```

# Eval (Tracking)

1. Save detection results. Specify resolution `${RES}` and number of GPUs `${NGPU}`. We set default model checkpoint for evaluation. Change it by `--checkpoint`. By default, results will be saved to `mmtracking/track_pkl`, controlled by argument `output_dir`.
    ```
    # conda activate alphachimp
    python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 \
        --nproc_per_node=${NGPU} --master_port=25525 tools/save_tracking.py \
        configs/alphachimp/alphachimp_tracking${RES}.py \
        --checkpoint work_dirs/alphachimp/alphachimp_res${RES}.pth \
        --gpus ${NGPU}
    ```

2. Download [fixed annotation file](https://drive.google.com/file/d/1usYe3gQ6SQfazJAx8UDR5vql7dZfAMv-/view?usp=drive_link) and place it under `data/ChimpACT_processed/annotations/test_fix.json`. 

3. We evaluate tracking performance in a new environment. Create tracking evaluation environment by
    ```
    # go inside mmtracking
    cd mmtracking

    conda create -n eval_tracking python=3.8
    conda activate eval_tracking
            
    conda install ffmpeg
    pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
    pip install --no-deps -r requirements/alphachimp.txt   # no worry if mmcv cannot be successfully installed at this step
    pip install -v -e .

    pip install openmim
    mim install mmcv-full==1.7.2
    
    cd TrackEval
    pip install -v -e .
    cd ..
    ```

4. Evaluation on tracking performance. This will load the saved file `mmtracking/track_pkl/summary.pkl` generated in previous step.
    ```
    # conda activate eval_tracking
    python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=1 --master_port=25525 \
        tools/evaluate_tracking.py configs/evaluate_tracking.py
    ```

# Citation
```bibtex
@article{ma2024alphachimp,
    title={AlphaChimp: Tracking and Behavior Recognition of Chimpanzees},
    author={Ma, Xiaoxuan and Lin, Yutang and Xu, Yuan and Kaufhold, Stephan and Terwilliger, Jack and Meza, Andres and Zhu, Yixin and Rossano, Federico and Wang, Yizhou},
    journal={arXiv preprint arXiv:2410.17136},
    year={2024}
}
```

# Acknowledgement
This repo is built on the excellent work [MMDetection](https://github.com/open-mmlab/mmdetection), [MMTracking](https://github.com/open-mmlab/mmtracking), and [MMAction2](https://github.com/open-mmlab/mmaction2). Thanks for these great projects.
