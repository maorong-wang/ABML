# Bridging the Capacity Gap for Online Knowledge Distillation
This repo is the official Implementation of our paper "Bridging the Capacity Gap for Online Knowledge Distillation". This paper is accepted by IEEE MIPR 2023.

## Environment

- Python 3.6
- PyTorch 1.9.0
- torchvision 0.10.0
  
Install the package:
```bash
pip3 install -r requirements.txt
```

## Training
- Define hyperparameters in a yaml configuration file, you can use `example.yaml` as a template.
- Run the command:
  `python train.py -cfg *path-to-config-file*`

## Acknowledgement
Credit to [mdistiller](https://github.com/megvii-research/mdistiller) for the codebase.

## Cite
If you found our work useful, please cite us:
```
@inproceedings{wang2023bridging,
  title={Bridging the Capacity Gap for Online Knowledge Distillation},
  author={Wang, Maorong and Yu, Hao and Xiao, Ling and Yamasaki, Toshihiko},
  booktitle={2023 IEEE 6th International Conference on Multimedia Information Processing and Retrieval (MIPR)},
  pages={1--4},
  year={2023},
  organization={IEEE}
}
```
