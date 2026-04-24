# SPAC: Stable pseudo-label refinement and action completeness learning for weakly supervised temporal action localization

Han Wen, Guangping Zeng, Qingchuan Zhang, Shuo Yang, Yupeng Hou, Qicheng Ma


## Recommended Environment

* Python version: 3.8.0
* Pytorch version: 2.3.1
* CUDA version: 12.1
* Tensorboard version: 2.14.0


## Data Preparation
1. Prepare [THUMOS14](https://www.crcv.ucf.edu/THUMOS14/) dataset.
    * To help you better reproduce our results, we recommend using the [pretrained I3D model](https://github.com/Finspire13/pytorch-i3d-feature-extraction.git) for feature extraction, as we did.
    * You can also get access of it from [Google Drive](https://drive.google.com/drive/folders/1_fGZpPM0PCTAgGQbQpBQEhK2KculypEu?usp=drive_link).

2. Prepare proposals.
    * You can just download the proposals used in our paper from [Google Drive](https://drive.google.com/drive/folders/13iuiiz4xlbAmCMZCwH1xVxPs_meSHoCy?usp=drive_link).

3. Place the features and annotations inside a `data/Thumos14reduced/` folder,  proposals inside a `proposals` folder and descriptors inside a `descriptors` folder. Make sure the data structure is as below.

```
    ├── data
        └── Thumos14reduced
            ├── Thumos14reduced-I3D-JOINTFeatures.npy
            └── Thumos14reduced-Annotations
                ├── Ambiguous_test.txt
                ├── classlist.npy
                ├── duration.npy
                ├── extracted_fps.npy
                ├── labels_all.npy
                ├── labels.npy
                ├── original_fps.npy
                ├── segments.npy
                ├── subset.npy
                └── videoname.npy
    ├── proposals4Thumos14
        ├── Proposals_Thumos14reduced_train.json
        ├── Proposals_Thumos14reduced_test.json
    ├── descriptors
        └── Thumos14reduced
            ├── general_appearance_descriptors.npy
            ├── general_motion_descriptors.npy
```

## Quick Start

### Training

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --run_type train
```

## Acknowledgement

We referred to the following repos when writing our code. We sincerely thank them for their outstanding contributions to the open-source community!

- [W-TALC](https://github.com/sujoyp/wtalc-pytorch)
- [CO2-Net](https://github.com/harlanhong/MM2021-CO2-Net)
- [P-MIL](https://github.com/RenHuan1999/CVPR2023_P-MIL)
- [SEAL_WTAL](https://github.com/Lkydong2020/SEAL_Wtal)
