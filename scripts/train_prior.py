import argparse
import pickle
import os
import torch
from torch.utils.data import DataLoader
from dalle2_pytorch import DiffusionPriorNetwork, DiffusionPrior, DiffusionPriorTrainer
from tqdm import tqdm

import sys
sys.path.append('..')
from src.data.singlemodal_dataset import *
from src.data.multimodal_dataset import *
from src.utils import *
from src.model_config import config_dict
from src.models.order import OrderModel
from src.models import OrderModel
from src.utils import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=200)

    parser.add_argument("--log", type=str, default='log')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", type=str, default='composite', choices=['composite','fiber'])

    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--backbone", type=str, default='CLIP_ViT-B/16')
    parser.add_argument("--config", type=str, default='order')
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--prefix", type=str, default='order_dyn', choices=['order_dyn','order_alpha'])
    parser.add_argument("--n_threads", type=int, default=2)
    args = parser.parse_args()

    if 'alpha' in args.prefix:
        setting = f'seed{args.seed}_Alpha{str(args.alpha)}_{args.dataset}'
    else:
        setting = f'seed{args.seed}_{args.dataset}'
    args.savepth = os.path.join('save', args.prefix, args.backbone.replace('/','_'), setting)
    if not os.path.exists(args.savepth):
        raise RuntimeError(f'path {args.savepth} not exists')
    args.logger = create_logger(args.savepth, f'{args.log}-prior')
    args.weightpth = os.path.join(args.savepth, "weight-final.pth")
    args.setting = setting
    args.logger.info(print_arg(args))

    return args


def get_model(args, device):
    model_cls = OrderModel
    config_sgpt = config_dict[args.config]
    config_prior = config_dict['prior']
    extractor = model_cls(
        cond_dim=args.cond_dim,
        hidden_dim=config_sgpt["hidden_dim"],
        common_dim=config_sgpt["common_dim"],
        latent_dim=config_sgpt["latent_dim"],
        dropout=config_sgpt["dropout"],
        backbone=args.backbone,
        lora_r=args.r,
        cardinality=args.cardin
    ).to(device)
    extractor.load_state_dict(torch.load(args.weightpth), strict=False)
    extractor.eval()

    prior_network = DiffusionPriorNetwork(
        dim=config_prior["dim"],
        depth=config_prior["depth"],
        dim_head=config_prior["dim_head"],
        heads=config_prior["heads"]
    ).to(device)

    diffusion_prior = DiffusionPrior(
        net=prior_network,
        image_embed_dim=config_prior["image_embed_dim"],
        timesteps=config_prior["timesteps"],
        cond_drop_prob=config_prior["cond_drop_prob"],
        condition_on_text_encodings=False
    ).to(device)

    return extractor, diffusion_prior


def main(args):
    g = torch.Generator()
    g.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.dataset == 'composite':
        feature_cols = ['NumFibers','MMA','Vf','A11','A12','A13','A22','A23','A33']
        target_cols = ['Yield strength','Elongation']
        cate_cols = None
        idx_col = 'Image index'
        image_dir = '../datasets_composite/processed'
        trainfile = '../datasets_composite/train.csv'
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
        trainfile = '../datasets_fiber/table/mech/train.csv'
        dataset_cls = MultiModalFibreDataset
        cardin = [2]
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),  
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),  
            Rotate90(),
            transforms.ToTensor(),  
        ])
        use_normalize = True
    else:
        raise NotImplementedError(f'dataset {args.dataset} unknown')
    args.cardin = cardin
    args.cond_dim = len(feature_cols)
    train_set = dataset_cls(csv_file=trainfile, image_dir=image_dir, feature_cols=feature_cols, target_cols=target_cols, extracted_fea=None, istrain=True, train_transform=transform_train, test_transform=transform_test, idx_col=idx_col, scaler=None, category_cols=cate_cols, use_normalize=use_normalize)

    extractor, diffusion_prior = get_model(args, device)
    diffusion_prior_trainer = DiffusionPriorTrainer(
        diffusion_prior,
        lr=args.lr,
        wd=args.weight_decay,
        ema_beta=0.99,
        ema_update_after_step=1000,
        ema_update_every=10,
    ).to(device)

    step = 0
    batch_loss_accumulator = []  # To track losses for every 10 steps
    for epoch in range(args.n_epochs):
        args.logger.info(f'Epoch {epoch}:')
        loader = DataLoader(train_set, batch_size=128, shuffle=True)
        for idx, (_, _, batched_table, batched_images) in enumerate(loader):
            batched_table = batched_table.to(device)
            batched_images = batched_images.to(device)

            table_emb = extractor.encode(batched_table, 'tab').detach()
            images_emb = extractor.encode(batched_images, 'image').detach()

            loss = diffusion_prior_trainer(
                text_embed=table_emb,
                image_embed=images_emb,
            )
            diffusion_prior_trainer.update()

            batch_loss_accumulator.append(loss)
            step += 1
            if step % 10 == 0:
                avg_loss = sum(batch_loss_accumulator) / len(batch_loss_accumulator)
                args.logger.info(f"Step {step}: Average Loss = {avg_loss:.4f}")
                batch_loss_accumulator = []
    savepth = os.path.join(args.savepth, f"prior.pth")
    diffusion_prior_trainer.save(savepth)


if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)
    main(args)
