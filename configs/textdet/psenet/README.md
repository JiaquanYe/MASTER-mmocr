# PSENet

## Introduction

[ALGORITHM]

```bibtex
@inproceedings{wang2019shape,
  title={Shape robust text detection with progressive scale expansion network},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Hou, Wenbo and Lu, Tong and Yu, Gang and Shao, Shuai},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9336--9345},
  year={2019}
}
```

## Results and models

### CTW1500

|                                Method                                | Backbone | Extra Data | Training set  |   Test set   | #epochs | Test size | Recall | Precision | Hmean |                                                                                                Download                                                                                                |
| :------------------------------------------------------------------: | :------: | :--------: | :-----------: | :----------: | :-----: | :-------: | :----: | :-------: | :---: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [PSENet-4s](/configs/textdet/psenet/psenet_r50_fpnf_600e_ctw1500.py) | ResNet50 |     -      | CTW1500 Train | CTW1500 Test |   600   |   1280    | 0.728  |   0.849   | 0.784 | [model](https://download.openmmlab.com/mmocr/textdet/psenet/psenet_r50_fpnf_600e_ctw1500_20210401-216fed50.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/psenet/20210401_215421.log.json) |

### ICDAR2015

|                                 Method                                 | Backbone |                                                                Extra Data                                                                 | Training set | Test set  | #epochs | Test size | Recall | Precision | Hmean |                                                                                            Download                                                                                             |
| :--------------------------------------------------------------------: | :------: | :---------------------------------------------------------------------------------------------------------------------------------------: | :----------: | :-------: | :-----: | :-------: | :----: | :-------: | :---: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [PSENet-4s](/configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015.py) | ResNet50 |                                                                     -                                                                     |  IC15 Train  | IC15 Test |   600   |   2240    | 0.784  |   0.831   | 0.807 | [model](https://download.openmmlab.com/mmocr/textdet/psenet/psenet_r50_fpnf_600e_icdar2015-c6131f0d.pth) \| [log](https://download.openmmlab.com/mmocr/textdet/psenet/20210331_214145.log.json) |
| [PSENet-4s](/configs/textdet/psenet/psenet_r50_fpnf_600e_icdar2015.py) | ResNet50 | pretrain on IC17 MLT [model](https://download.openmmlab.com/mmocr/textdet/psenet/psenet_r50_fpnf_600e_icdar2017_as_pretrain-3bd6056c.pth) |  IC15 Train  | IC15 Test |   600   |   2240    | 0.834  |   0.861   | 0.847 |                                  [model](https://download.openmmlab.com/mmocr/textdet/psenet/psenet_r50_fpnf_600e_icdar2015_pretrain-eefd8fe6.pth) \| [log]()                                   |
