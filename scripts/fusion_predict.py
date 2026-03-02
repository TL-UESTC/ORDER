import argparse
import pickle
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
import os
import sys
sys.path.append('..')
from src.data.singlemodal_dataset import *
from src.data.multimodal_dataset import *
from src.model_config import config_dict
from src.models.order import OrderModel
from src.models.mlp import MechModel, FusionModel
from src.trainer.evaluator import Evaluator
from src.trainer.finetune_trainer import Trainer, FusionTrainer
from src.trainer.result_tracker import Result_Tracker
from src.trainer.scheduler import PolynomialDecayLR
from src.utils import *


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=100)

    parser.add_argument("--dataset", type=str, default='composite', choices=['composite','fiber'])
    parser.add_argument("--metric", type=str, default='rmse')

    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=5e-4)

    parser.add_argument("--log", type=str, default='log')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_threads", type=int, default=2)

    parser.add_argument("--backbone", type=str, default='CLIP_ViT-B/16')
    parser.add_argument("--config", type=str, default='order')
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--prefix", type=str, default='order_dyn', choices=['order_dyn','order_alpha'])
    args = parser.parse_args()
    if 'alpha' in args.prefix:
        setting = f'seed{args.seed}_Alpha{str(args.alpha)}_{args.dataset}'
    else:
        setting = f'seed{args.seed}_{args.dataset}'
    args.savepth = os.path.join('save', args.prefix, args.backbone.replace('/','_'), setting)
    if not os.path.exists(args.savepth):
        raise RuntimeError(f'path {args.savepth} not exists')
    args.logger = create_logger(args.savepth, f'{args.log}-predict_w_fusion')
    args.weightpth = os.path.join(args.savepth, "weight-final.pth")
    args.setting = setting
    args.logger.info(print_arg(args))
    return args


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.dataset == 'composite':
        feature_cols = ['NumFibers','MMA','Vf','A11','A12','A13','A22','A23','A33']
        target_cols = ['Yield strength','Elongation']
        cate_cols = None
        idx_col = 'Image index'
        image_dir = '../datasets_composite/processed'
        trainfile = '../datasets_composite/train_pred.csv'
        testfile = '../datasets_composite/test_pred.csv'
        valfile = '../datasets_composite/val_pred.csv'
        transform_train, transform_test = None, None
        dataset_cls = MultiModalCompositeDataset
        cardin = []
        use_normalize = False
    elif args.dataset == 'fiber':
        feature_cols = ['f','c','v','r','t','w','dir']
        target_cols = ['fracture','elongation','elastic modulus','tangent modulus','yield']
        cate_cols = ['dir']
        idx_col = 'ID'
        image_dir = '../datasets_fiber/images/preprocessed'
        trainfile = '../datasets_fiber/table/mech/train_pred.csv'
        testfile = '../datasets_fiber/table/mech/test_pred.csv'
        valfile = '../datasets_fiber/table/mech/val_pred.csv'
        dataset_cls = MultiModalFibreDataset
        cardin = [2]
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ToTensor(),  
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.RandomVerticalFlip(p=0.1),
            Rotate90(),
            transforms.ToTensor(),  
        ])
        use_normalize = True
    else:
        raise NotImplementedError(f'dataset {args.dataset} unknown')
    train_set = dataset_cls(csv_file=trainfile, image_dir=image_dir, feature_cols=feature_cols, target_cols=target_cols, extracted_fea=None, istrain=True, train_transform=transform_train, test_transform=transform_test, idx_col=idx_col, scaler=None, category_cols=cate_cols, use_normalize=use_normalize)
    test_set = dataset_cls(csv_file=testfile, image_dir=image_dir, feature_cols=feature_cols, target_cols=target_cols, extracted_fea=None, istrain=False, train_transform=transform_train, test_transform=transform_test, idx_col=idx_col, scaler=train_set.scaler, category_cols=cate_cols, use_normalize=use_normalize)
    val_set = dataset_cls(csv_file=valfile, image_dir=image_dir, feature_cols=feature_cols, target_cols=target_cols, extracted_fea=None, istrain=False, train_transform=transform_train, test_transform=transform_test, idx_col=idx_col, scaler=train_set.scaler, category_cols=cate_cols, use_normalize=use_normalize)

    train_mean, train_std = np.array(train_set.mean), np.array(train_set.std)
    cond_dim = len(feature_cols)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    model_cls = OrderModel
    trainer_cls = FusionTrainer 
    config_sgpt = config_dict[args.config]
    extractor = model_cls(
        cond_dim=cond_dim,
        hidden_dim=config_sgpt["hidden_dim"],
        common_dim=config_sgpt["common_dim"],
        latent_dim=config_sgpt["latent_dim"],
        dropout=config_sgpt["dropout"],
        backbone=args.backbone,
        lora_r=args.r,
        cardinality=cardin
    ).to(device)
    extractor.load_state_dict(torch.load(args.weightpth), strict=False)
    n_task = len(target_cols)
    model = FusionModel(
        extractor,
        latent_dim=config_sgpt["latent_dim"],
        hidden_dim=config_sgpt["hidden_dim"],
        out_dim=n_task,
        num_layers=3,
        dropout=args.dropout,
        modality='mix'
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(optimizer, warmup_updates=args.n_epochs * len(train_set) // 32 // 10, tot_updates=args.n_epochs * len(train_set) // 32, lr=args.lr, end_lr=1e-9, power=1)

    loss_fn = MSELoss()
    evaluator = Evaluator("MechModel", args.metric, n_task, mean=train_mean, std=train_std)
    final_evaluator_1 = Evaluator("MechModel", "rmse_split", n_task, mean=train_mean, std=train_std)
    final_evaluator_2 = Evaluator("MechModel", "r2", n_task, mean=train_mean, std=train_std)
    result_tracker = Result_Tracker(args.metric)
    summary_writer = None

    trainer = trainer_cls(args, optimizer, lr_scheduler, loss_fn, evaluator, final_evaluator_1, final_evaluator_2, result_tracker, summary_writer, device=device, model_name='MechModel', label_mean=train_set.mean.to(device) if train_set.mean is not None else None, label_std=train_set.std.to(device) if train_set.std is not None else None)
    best_train, best_val, best_test, test_final = trainer.fit(model, train_loader, val_loader, test_loader)
    args.logger.info(f"train: {best_train:.3f}, val: {best_val:.3f}, test: {best_test:.3f}")

    for i in range(len(test_final[0])):
        args.logger.info(f"{target_cols[i]}: test rmse: {test_final[0][i]:.3f}\ttest r2: {test_final[1][i]:.3f}")


if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)
    main(args)
