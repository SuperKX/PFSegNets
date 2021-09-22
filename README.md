# Brief
Forked from [lxtGH/PFSegNets](https://github.com/lxtGH/PFSegNets), this page to record configuration info.    
Paper [cvpr-version](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_PointFlow_Flowing_Semantics_Through_Points_for_Aerial_Image_Segmentation_CVPR_2021_paper.pdf)/[arxiv-version](https://arxiv.org/pdf/2103.06564v1.pdf)

# Enviorenmet Config
This version cfged in Win10, I use anaconda to create virtual env. Installing cuda11 & corresponding cudnn version before is necessary.
```
conda create -n pfsegnet python=3.7.6
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
other libs are needed:
```
pip install opencv-python
pip install natsort
pip install tqdm
pip install scikit-image
pip install scipy
pip install tensorboard
pip install tensorboardX
```

# DataSet preparation
### 1 Download dataset
 [Potsdam](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-potsdam/) and
 [Vahihigen](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-vaihingen/) dataset. ( [iSAID](https://captain-whu.github.io/iSAID/) didn't be used here.)
 
### 2 Select data
I chose 2_Ortho_RGB.zip and 5_Labels_all.zip of Potsdam to do experiments, place them into the folder 'PFSegNets/dataset/'.
The data is organized in the following format: 
```
/dataset/
    └── train/
        └── images/
            ├── top_potsdam_2_10_RGB.tif
            ├── top_potsdam_2_11_RGB.tif 
            ...
        └── masks/
            ├── top_potsdam_2_10_label.tif
            ├── top_potsdam_2_11_label.tif
            ...
    └── val/
        ...
    └── test/
        ...
```
🔥 /val & /test have the same structures as /train .    
🔥 I chose top_potsdam_4_13_RGB, top_potsdam_6_12_RGB and top_potsdam_7_8_RGB as verification data, and top_potsdam_3_13_RGB, top_potsdam_5_12_RGB as test data.

### 3 convert labels to graymask
```
python convert_isprs_mask2graymask.py --phase train
python convert_isprs_mask2graymask.py --phase val
python convert_isprs_mask2graymask.py --phase test
```

### 4 split imgs into patches    
```
python split_isprs.py
```
🔥 remember to change the paths to yours.

### 5 change filepaths in `config.py`
```
__C.DATASET.POSDAM_DIR = #yourpath#
``` 


# Pretrained Models

Baidu Pan Link: https://pan.baidu.com/s/1MWzpkI3PwtnEl1LSOyLrLw  4lwf     
Google Drive Link: https://drive.google.com/drive/folders/1C7YESlSnqeoJiR8DWpmD4EVWvwf9rreB?usp=sharing

change the model paths of `network/resnet_d.py`

# Model Checkpoints

  <table><thead><tr><th>Dataset</th><th>Backbone</th><th>mIoU</th><th>Model</th></tr></thead><tbody>
<tr><td>iSAID</td><td>ResNet50</td><td>66.9</td><td><a href="https://drive.google.com/file/d/1igB0y-5IybcIxf0cALFoqh0Pg36OxWR-/view?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a>&nbsp;|&nbsp;<a href="https://pan.baidu.com/s/1xX2DXdQ5SdpKA3w2EAdZUA" target="_blank" rel="noopener noreferrer">Baidu Pan</a>(v3oj)</td></tr>
<tr><td>Potsdam</td><td>ResNet50</td><td>75.4</td><td><a href="https://drive.google.com/file/d/1tVvPLaMLBp55HfyDhRgmRcMOW44CSc6s/view?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a>&nbsp;|&nbsp;<a href="https://pan.baidu.com/s/1NX1k80NBIrA_G03AsmzZ1w" target="_blank" rel="noopener noreferrer">Baidu Pan</a>(lhlf)</td></tr>
<tr><td>Vaihigen</td><td>ResNet50</td><td>70.4</td><td><a href="https://drive.google.com/file/d/1C3FrXPo8-LuBGUJcC6PCcMP-FP8zVXXb/view?usp=sharing" rel="noopener noreferrer">Google Drive</a>&nbsp;|&nbsp;<a href="https://pan.baidu.com/s/1LSOViE817pS2XpzMPCBbwA" target="_blank" rel="noopener noreferrer">Baidu Pan</a>(54qm)</td></tr>
</tbody></table>

# Evaluation
```
python eval.py --dataset Posdam --arch network.pointflow_resnet_with_max_avg_pool.DeepR50_PF_maxavg_deeply --inference_mode  whole --single_scale --scales 1.0 --split test --cv_split 0 --avgpool_size 9 --edge_points 128 --match_dim 64 --resize_scale 896 --mode semantic --no_flip --ckpt_path CHECKPOINTPATH --snapshot SAVERESULTS
```
🔥 'CHECKPOINTPATH' is checkpoints path, eg: E:\PFSegNets\checkpoints\pfnet_r50_postdam.pth    
🔥 'SAVERESULTS' is results path, eg: E:\PFSegNets\results    


For example, when evaluating PFNet on validation set of iSAID dasaset:
```bash
sh scripts/pointflow/test/test_iSAID_pfnet_R50.pth path_to_checkpoint path_to_save_results
```
If you want to save images during evaluating, add args: `dump_images`, which will take more time.

# Training

To be note that, our models are trained on 8 V-100 GPUs with 32GB memory.
 **It is hard to reproduce such best results if you do not have such resources.**
For example, when training PFNet on iSAID dataset:
```bash
sh scripts/pointflow/train_iSAID_pfnet_r50.sh
```

# Citation
If you find this repo is helpful to your research. Please consider cite our work.

```
@inproceedings{li2021pointflow,
  title={PointFlow: Flowing Semantics Through Points for Aerial Image Segmentation},
  author={Li, Xiangtai and He, Hao and Li, Xia and Li, Duo and Cheng, Guangliang and Shi, Jianping and Weng, Lubin and Tong, Yunhai and Lin, Zhouchen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4217--4226},
  year={2021}
}
```

# Acknowledgement
This repo is based on NVIDIA segmentation [repo](https://github.com/NVIDIA/semantic-segmentation). 
We fully thank their open-sourced code.
