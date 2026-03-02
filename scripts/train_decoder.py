import argparse
import torch
import os
from torch.utils.data import DataLoader
from dalle2_pytorch import Unet, Decoder, DecoderTrainer
from torchvision.utils import save_image
from tqdm import tqdm

import sys
sys.path.append('..')
from src.data.singlemodal_dataset import *
from src.data.multimodal_dataset import *
from src.utils import *
from src.models.order import OrderModel
from src.data.gen_dataset import ImageDataset
from src.model_config import config_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=50)

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
    args.logger = create_logger(args.savepth, f'{args.log}-decoder')
    args.weightpth = os.path.join(args.savepth, "weight-final.pth")
    args.setting = setting
    args.logger.info(print_arg(args))
    return args


def get_model(args, device):
    model_cls = OrderModel
    config_sgpt = config_dict[args.config]
    config_decoder = config_dict['decoder']
    extractor = model_cls(
        cond_dim=args.cond_dim,
        hidden_dim=config_sgpt["hidden_dim"],
        common_dim=config_sgpt["common_dim"],
        latent_dim=config_sgpt["latent_dim"],
        dropout=config_sgpt["dropout"],
        backbone=args.backbone,
        lora_r=args.r,
        cardinality=args.cardin
    ).cuda()
    extractor.load_state_dict(torch.load(args.weightpth, map_location='cpu'), strict=False)
    extractor.eval()

    unet = Unet(
        dim=config_decoder['dim'],
        image_embed_dim=config_decoder['image_embed_dim'],
        cond_dim=config_decoder['cond_dim'],
        channels=config_decoder['channels'],
        dim_mults=config_decoder['dim_mults']
    ).cuda()

    decoder = Decoder(
        unet=unet,
        image_size=config_decoder['image_size'],
        timesteps=config_decoder['timesteps'],
        image_cond_drop_prob=config_decoder['image_cond_drop_prob'],
        text_cond_drop_prob=config_decoder['text_cond_drop_prob'],
        learned_variance=False
    ).cuda()

    return extractor, decoder


def main(args):
    g = torch.Generator()
    g.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.dataset == 'fiber':
        cond_dim = 7
        cardin = [2]
        train_set = ImageDataset(root_path='../datasets_fiber', dataset='gen')
    elif args.dataset == 'composite':
        rootpath = '../datasets_composite'
        tar_col = ['Yield strength','Elongation']
        fea_col = ['NumFibers','MMA','Vf','A11','A12','A13','A22','A23','A33']
        cardin = []
        cond_dim = len(fea_col)
        train_set = CompositeImageDataset('train', rootpath, os.path.join(rootpath, 'processed'), feature_cols=fea_col, target_cols=tar_col)
    else:
        raise NotImplementedError
    args.cardin = cardin
    args.cond_dim = cond_dim
    loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=args.n_threads, drop_last=True)
    extractor, decoder = get_model(args, device)
    decoder_trainer = DecoderTrainer(
        decoder,
        lr=args.lr,
        wd=args.weight_decay,
        ema_beta=0.99,
        ema_update_after_step=1000,
        ema_update_every=10,
    ).cuda()

    decoder.train()
    step = 0
    batch_loss_accumulator = []  # To track losses for every 100 steps

    for epoch in range(args.n_epochs):
        args.logger.info(f'Epoch {epoch}')
        for i, data in enumerate(loader):
            if args.dataset == 'composite':
                batched_images = data[0].cuda()
            else:
                batched_images = data.cuda()
            images_emb = extractor.encode(batched_images, 'image').detach()

            loss = decoder_trainer(
                batched_images,
                image_embed=images_emb,
            )
            decoder_trainer.update(1)

            batch_loss_accumulator.append(loss)
            step += 1
            if step % 1000 == 0:
                avg_loss = sum(batch_loss_accumulator) / len(batch_loss_accumulator)
                args.logger.info(f"Step {step}: Average Loss = {avg_loss:.4f}")
                batch_loss_accumulator = []

            if step % 2000 == 0:
                sample = decoder_trainer.sample(image_embed=images_emb[0].unsqueeze(0))
                save_image(sample, os.path.join(args.savepth, f'gen_{epoch}_{step}.png'))

                savepth = os.path.join(args.savepth, f"decoder.pth")
                decoder_trainer.save(savepth)


if __name__ == "__main__":
    args = parse_args()
    main(args)
