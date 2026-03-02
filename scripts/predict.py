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
from src.models.mlp import MechModel
from src.trainer.evaluator import Evaluator
from src.trainer.finetune_trainer import Trainer
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
    parser.add_argument("--modal", type=str, default='tab', choices=['tab','image'])
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
    args.logger = create_logger(args.savepth, f'{args.log}-predict_w_{args.modal}')
    args.weightpth = os.path.join(args.savepth, "weight-final.pth")
    args.setting = setting
    args.logger.info(print_arg(args))
    return args


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.dataset == 'fiber':
        feature_cols = ['f','c','v','r','t','w','dir']
        cond_dim = 7
        n_task = 5
        cardin = [2]
        tar_col = ['fracture','elongation','elastic modulus','tangent modulus','yield']
        data_path = '../datasets_fiber/'
        rootpath = '../datasets_fiber/table/mech'
        image_dir = '../datasets_fiber/images/preprocessed'
        if args.modal == 'tab':
            train_set = TableDataset(split="mech", dataset_type="train_pred", root_path=data_path, scaler=None)
            val_set = TableDataset(split="mech", dataset_type="val_pred", root_path=data_path, scaler=train_set.scaler)
            test_set = TableDataset(split="mech", dataset_type="test_pred", root_path=data_path, scaler=train_set.scaler)
        else:
            train_set = FibreImageDataset('train_pred', rootpath, image_dir, feature_cols=feature_cols, target_cols=tar_col, id_col='ID')
            test_set = FibreImageDataset('test_pred', rootpath, image_dir, feature_cols=feature_cols, target_cols=tar_col, id_col='ID')
            val_set = FibreImageDataset('val_pred', rootpath, image_dir, feature_cols=feature_cols, target_cols=tar_col, id_col='ID')
    elif args.dataset == 'composite':
        rootpath = '../datasets_composite'
        tar_col = ['Yield strength','Elongation']
        fea_col = ['NumFibers','MMA','Vf','A11','A12','A13','A22','A23','A33']
        cate_cols = None
        cardin = []
        cond_dim = len(fea_col)
        n_task = 2
        if args.modal == 'tab':
            train_set = CompositeTableDataset('train_pred', rootpath, fea_col, tar_col, cate_cols, None)
            test_set = CompositeTableDataset('test_pred', rootpath, fea_col, tar_col, cate_cols, train_set.scaler)
            val_set = CompositeTableDataset('val_pred', rootpath, fea_col, tar_col, cate_cols, train_set.scaler)
        elif args.modal == 'image':
            train_set = CompositeImageDataset('train_pred', rootpath, os.path.join(rootpath, 'processed'), feature_cols=fea_col, target_cols=tar_col)
            test_set = CompositeImageDataset('test_pred', rootpath, os.path.join(rootpath, 'processed'), feature_cols=fea_col, target_cols=tar_col)
            val_set = CompositeImageDataset('val_pred', rootpath, os.path.join(rootpath, 'processed'), feature_cols=fea_col, target_cols=tar_col)
    train_mean, train_std = train_set.mean.numpy(), train_set.std.numpy()

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    model_cls = OrderModel
    trainer_cls = Trainer 
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

    model = MechModel(
        extractor,
        latent_dim=config_sgpt["latent_dim"],
        hidden_dim=config_sgpt["hidden_dim"],
        out_dim=n_task,
        num_layers=3,
        dropout=args.dropout,
        modality=args.modal
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
        args.logger.info(f"{tar_col[i]}: test rmse: {test_final[0][i]:.3f}\ttest r2: {test_final[1][i]:.3f}")


if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)
    main(args)
