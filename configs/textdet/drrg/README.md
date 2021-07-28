# DRRG

## Introduction

[ALGORITHM]

```bibtex
@article{zhang2020drrg,
  title={Deep relational reasoning graph network for arbitrary shape text detection},
  author={Zhang, Shi-Xue and Zhu, Xiaobin and Hou, Jie-Bo and Liu, Chang and Yang, Chun and Wang, Hongfa and Yin, Xu-Cheng},
  booktitle={CVPR},
  pages={9699-9708},
  year={2020}
}
```

## Results and models

### CTW1500

|                              Method                              | Pretrained Model | Training set  |   Test set   | #epochs | Test size | Recall | Precision | Hmean |                                                                                  Download                                                                                              |
| :--------------------------------------------------------------: | :--------------: | :-----------: | :----------: | :-----: | :-------: | :----: | :-------: | :---: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [DRRG](/configs/textdet/drrg/drrg_r50_fpn_unet_1200e_ctw1500.py) |     ImageNet     | CTW1500 Train | CTW1500 Test |  1200   |    640    | 0.822  |   0.858   | 0.840 | [model](https://download.openmmlab.com/mmocr/textdet/drrg/drrg_r50_fpn_unet_1200e_ctw1500-1abf4f67.pth) \ [log](https://download.openmmlab.com/mmocr/textdet/drrg/20210511_234719.log) |
