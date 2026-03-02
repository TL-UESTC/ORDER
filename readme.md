## Pytorch implementation for *[Beyond Alignment: Learning Ordinal-Aware Image-Tabular Representations for Composite Materials](https://arxiv.org/abs/2602.02513)*


## Create environment
```
conda env create -f environment.yml
conda activate order
cd scripts
```

## Datasets
- Our code supports the `composite` dataset proposed in this paper, and the `fiber` dataset provided by MatMCL.
- To run with fiber dataset:
  1. Download the dataset following https://github.com/wuyuhui-zju/MatMCL to the root folder  and rename it as `datasets_fiber` (/ORDER/datasets_fiber).
  2. **Replace** all csv files under `datasets_fiber/table/mech` with the csv files provided in folder `dataset_fiber_table-mech`, which contains the data splits necessary for implementing ORDER.
- Our composite dataset is available upon request, please check DATA AVAILABILITY section in paper to address the request.


## Pretrain
- Pretrain an ORDER-dyn model with the following command:
```
python train_order_dyn.py --dataset [composite/fiber] --mode train
```
Command `--dataset` decides the Composite or Nanofiber dataset to use. Check `scripts/save/order_dyn` for results and log.
- Pretrain an ORDER- $\alpha$ model with the following command:
```
python train_order_alpha.py --dataset [composite/fiber] --alpha [0.2/0.5/...] --mode train
```
The `--alpha` command controls the $\alpha$ parameter used (specify `--alpha 0.0` for the 'CMCL' baseline). Check `scripts/save/order_alpha` for results and log.

## Pretrain
- Pretrain an ORDER-dyn model with the following command:
```
python train_order_dyn.py --dataset [composite/fiber] --mode train
```
Command `--dataset` decides the Composite or Nanofiber dataset to use. Check `scripts/save/order_dyn` for results and log.
- Pretrain an ORDER- $\alpha$ model with the following command:
```
python train_order_alpha.py --dataset [composite/fiber] --alpha [0.2/0.5/...] --mode train
```
The `--alpha` command controls the $\alpha$ parameter used (specify `--alpha 0.0` for the 'CMCL' baseline). Check `scripts/save/order_alpha` for results and log.

## Cross-modal retrieval
- Retrieve using ORDER-dyn: 
```
python train_order_dyn.py --dataset [composite/fiber] --mode test
```
- Retrieve using ORDER- $\alpha$: 
```
python train_order_alpha.py --dataset [composite/fiber] --alpha [0.2/0.5/...] --mode test
```

## Property prediction
- Single-modality prediction:
```
python predict.py --prefix [order_alpha/order_dyn] --alpha [0.2/0.5/...] --dataset [composite/fiber] --modal [image/tab]
```
Command `--prefix` specifies the method used (ORDER- $\alpha$ or ORDER-dyn), `modal` specifies the modality used.
- Modality fusion prediction:
```
python fusion_predict.py --prefix [order_alpha/order_dyn] --alpha [0.2/0.5/...] --dataset [composite/fiber] 
```

## Microstructure generation
- Train prior:
```
python train_prior.py --prefix [order_alpha/order_dyn] --dataset [composite/fiber] --alpha [0.2/0.5/...]
```
- Train decoder:
```
python train_decoder.py --prefix [order_alpha/order_dyn] --dataset [composite/fiber] --alpha [0.2/0.5/...] --n_epochs [50/2000]
```
Please set `--n_epochs 2000` for Composite data, and `--n_epochs 50` for Nanofiber data.
- Generate images 
```
python generate.py --prefix [order_alpha/order_dyn] --dataset [composite/fiber] --alpha [0.2/0.5/...] --split [train/test]
```
Specify `--split train` for in-distribution image generation, and `--split test` for out-of-distribution image generation. Check generated images at `save/...../gen-train(gen-test)/`
- Evaluate generated images
```
python eval_generate.py --prefix [order_alpha/order_dyn] --dataset [composite/fiber] --alpha [0.2/0.5/...] --split [train/test]
```

**Note: Parameter `--alpha` can be omitted for all tasks with ORDER-dyn (`--prefix order_dyn`)**
