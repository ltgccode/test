# LTGC: Long-Tail Recognition via Leveraging Generated Content

## Dataset Preparation

### (1) Three bechmark datasets
- Please download these datasets and put them to the /data file.
- ImageNet-LT and Places-LT can be found at [here](https://drive.google.com/drive/u/0/folders/1j7Nkfe6ZhzKFXePHdsseeeGI877Xu1yf).
- iNaturalist data should be the 2018 version from [here](https://github.com/visipedia/inat_comp).

```
data
├── ImageNet_LT
│   ├── test
│   ├── train
│   └── val
├── Place365
│   ├── data_256
│   ├── test_256
│   └── val_256
└── iNaturalist 
    ├── test2018
    └── train_val2018
```

### (2) Txt files
```
data_txt
├── ImageNet_LT
│   ├── ImageNet_LT_test.txt
│   ├── ImageNet_LT_train.txt
│   └── ImageNet_LT_val.txt
├── Places_LT_v2
│   ├── Places_LT_test.txt
│   ├── Places_LT_train.txt
│   └── Places_LT_val.txt
└── iNaturalist18
    ├── iNaturalist18_train.txt
    ├── iNaturalist18_uniform.txt
    └── iNaturalist18_val.txt 
```

## Running Scripts
Before running, please replace your own OPENAI key.

### Generated Existing Tail-class Descriptions
``` bash
python lmm_i2t.py
```

### Generated Extended Tail-class Descriptions 
``` bash
python lmm_extension.py
```

### Generated Extended Data using Cyclic-assessing
``` bash
python draw_i2t.py
```