import argparse
import os
import torch
import sys
sys.path.append('..')
from src.data.singlemodal_dataset import *
from src.data.multimodal_dataset import *
from src.utils import *
from src.model_config import config_dict
from skimage.metrics import peak_signal_noise_ratio as psnr


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training")
    parser.add_argument("--seed", type=int, default=0)

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
    parser.add_argument("--split", type=str, default='train', choices=['train','test'])
    args = parser.parse_args()

    if 'alpha' in args.prefix:
        setting = f'seed{args.seed}_Alpha{str(args.alpha)}_{args.dataset}'
    else:
        setting = f'seed{args.seed}_{args.dataset}'
    args.savepth = os.path.join('save', args.prefix, args.backbone.replace('/','_'), setting)
    if not os.path.exists(args.savepth):
        raise RuntimeError(f'path {args.savepth} not exists')
    args.logger = create_logger(args.savepth, f'{args.log}-{args.split}-EvalGen')
    args.weightpth = os.path.join(args.savepth, "weight-final.pth")
    args.setting = setting
    args.logger.info(print_arg(args))

    return args


def null_sync(t, *args, **kwargs):
    return [t]


def get_image(path, grey=False):
    if not grey:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    ori_img = Image.open(path).convert('RGB')
    img = transform(ori_img)
    return img


def main(args):
    metrics = {}
    FID = {'feature': 2048}
    IS = {'feature': 'logits_unbiased'}
    KID = {'feature': 2048}
    LPIPS = {'net_type': 'alex'}
    g = torch.Generator()
    g.manual_seed(args.seed)
    output_dir = os.path.join(args.savepth, f'gen-{args.split}')
    filelis = os.listdir(output_dir)
    all_real, all_gen, all_gen_group = [], [], []
    for idx in filelis:
        cur_path = os.path.join(output_dir, idx)
        realimg_pth = os.path.join(cur_path, 'real.png')
        real_img = get_image(realimg_pth)
        gen_list = os.listdir(cur_path)
        cur_gen = []
        for gen in gen_list:
            if gen != 'real.png':
                cur_gen_path = os.path.join(cur_path, gen)
                all_gen.append(get_image(cur_gen_path, grey=True if args.dataset == 'composite' else False))
                cur_gen.append(get_image(cur_gen_path, grey=True if args.dataset == 'composite' else False))
        if len(all_gen_group) == 0 or len(cur_gen) == all_gen_group[-1].shape[0]:
            all_real.append(real_img)
            all_gen_group.append(torch.stack(cur_gen))  # [9, 3,224,224]

    all_gen_group = torch.stack(all_gen_group, 0).to(args.device) # [N,9,3,224,224]
    all_real = torch.stack(all_real, 0).to(args.device)
    all_gen = torch.stack(all_gen, 0).to(args.device)

    int_real_images = all_real.mul(255).add(0.5).clamp(0, 255).type(torch.uint8)
    int_generated_images = all_gen.mul(255).add(0.5).clamp(0, 255).type(torch.uint8)
    gen_num = int_generated_images.shape[0]
    btz = 1024

    fid = FrechetInceptionDistance(**FID, dist_sync_fn=null_sync).to(args.device)
    inception = InceptionScore(**IS, dist_sync_fn=null_sync).to(args.device)
    kernel_inception = KernelInceptionDistance(**KID, dist_sync_fn=null_sync, subset_size=int(len(all_real/2))).to(args.device)
    fid.update(int_real_images, real=True)
    kernel_inception.update(int_real_images, real=True)
    if gen_num >= btz:
        start, end = 0, btz
        while True:
            cur_gen_images = int_generated_images[start:end,:,:,:]
            fid.update(cur_gen_images, real=False)
            kernel_inception.update(cur_gen_images, real=False)
            inception.update(cur_gen_images)
            if end == gen_num:
                break
            start = end
            end = min(end+btz, gen_num)
    else:
        fid.update(int_generated_images, real=False)
        kernel_inception.update(int_generated_images, real=False)
        inception.update(int_generated_images)
    metrics["FID"] = fid.compute().item()

    is_mean, is_std = inception.compute()
    metrics["IS_mean"] = is_mean.item()
    metrics["IS_std"] = is_std.item()

    kid_mean, kid_std = kernel_inception.compute()
    metrics["KID_mean"] = kid_mean.item()
    metrics["KID_std"] = kid_std.item()

    metrics["LPIPS"] = 0
    metrics["PSNR"] = 0
    for i in range(all_gen_group.shape[1]):
        renorm_real_images = all_real.mul(2).sub(1).clamp(-1,1)
        cur_gen_group = all_gen_group[:,i,:,:,:].squeeze()
        renorm_generated_images = cur_gen_group.mul(2).sub(1).clamp(-1,1)
        lpips = LearnedPerceptualImagePatchSimilarity(**LPIPS, dist_sync_fn=null_sync).to(args.device)
        lpips.update(renorm_real_images, renorm_generated_images)
        cur_res = lpips.compute().item()
        metrics["LPIPS"] += cur_res / all_gen_group.shape[1]

        psnr_val = psnr(np.array(renorm_real_images.cpu()), np.array(renorm_generated_images.cpu()))
        metrics["PSNR"] += psnr_val / all_gen_group.shape[1]
    args.logger.info(f'{int_real_images.shape[0]} real, {int_generated_images.shape[0]} generated samples')
    args.logger.info(metrics)


if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)
    main(args)
