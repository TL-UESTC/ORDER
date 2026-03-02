import os
import random
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import colored
import sys
import logging
import functools
from sklearn.manifold import TSNE
import torchvision.transforms.functional
import cvxpy as cp
import cvxopt
import pandas as pd
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class EPO_LP(object):

    def __init__(self, m, eps=1e-3):
        cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"
        self.m = m
        self.eps = eps
        self.last_move = None
        self.C = cp.Parameter((m, m))   # C: Gradient inner products, G^T G
        self.Ca = cp.Parameter(m)       # d_bal^TG
        self.rhs = cp.Parameter(m)      # RHS of constraints for balancing

        self.alpha = cp.Variable(m)     # Variable to optimize

        obj_bal = cp.Maximize(self.alpha @ self.Ca)   # objective for balance
        constraints_bal = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Simplex
                           self.C @ self.alpha >= self.rhs]
        self.prob_bal = cp.Problem(obj_bal, constraints_bal)  # LP balance

        obj_dom = cp.Maximize(cp.sum(self.alpha @ self.C))  # obj for descent
        constraints_res = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Restrict
                           self.alpha @ self.Ca >= -cp.neg(cp.max(self.Ca)),
                           self.C @ self.alpha >= 0]
        constraints_rel = [self.alpha >= 0, cp.sum(self.alpha) == 1,  # Relaxed
                           self.C @ self.alpha >= 0]
        self.prob_dom = cp.Problem(obj_dom, constraints_res)  # LP dominance
        self.prob_rel = cp.Problem(obj_dom, constraints_rel)  # LP dominance

        self.gamma = 0     # Stores the latest Optimum value of the LP problem


    def get_alpha(self, G, G_val, loss_val,relax=False):
        assert len(G) == self.m, "length != m"
        self.C.value = G @ G.T
        self.Ca.value = G @ G_val

        rl = np.mean(G * G_val, axis=1)

        if loss_val.item() > self.eps:
            J = self.Ca.value > 0
            if len(np.where(J)[0]) > 0:
                J_star_idx = np.where(rl == np.max(rl))[0]
                self.rhs.value = self.Ca.value.copy()
                self.rhs.value[J] = -np.inf     # Not efficient; but works.
                self.rhs.value[J_star_idx] = 0
            else:
                self.rhs.value = np.zeros_like(self.Ca.value)
            self.gamma = self.prob_bal.solve(solver=cp.GLPK, verbose=False)
            self.last_move = "bal"
        else:
            if relax:
                self.gamma = self.prob_rel.solve(solver=cp.GLPK, verbose=False)
            else:
                self.gamma = self.prob_dom.solve(solver=cp.GLPK, verbose=False)
            self.last_move = "dom"

        return self.alpha.value


def compute_gradient_vector(loss, model, param_names=None):
    model.zero_grad()
    loss.backward(retain_graph=True)
    
    grads = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if param_names is None or name in param_names:
                grads.append(param.grad.data.clone().flatten())
    
    if len(grads) > 0:
        return torch.cat(grads)
    else:
        return None


class Rotate90:
    def __call__(self, img):
        return torchvision.transforms.functional.rotate(img, 90)


def print_arg(args):
    strr = ''
    arg_dict = vars(args)
    for k in arg_dict:
        strr += f'{k}: {arg_dict[k]}\n'
    return strr


@functools.lru_cache()
def create_logger(output_dir, file='1'):
    # create logger
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'{file}.log'), mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


def set_random_seed(seed=22, n_threads=16):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(n_threads)
    os.environ['PYTHONHASHSEED'] = str(seed)


def compute_retrieval(img_features, tab_features, k=1):
    img_norm = img_features.float()
    tab_norm = tab_features.float()
    similarity_matrix = torch.matmul(img_norm, tab_norm.T)
    labels = torch.arange(tab_norm.shape[0], device=similarity_matrix.device)

    _, img_to_tab_topk_idx = similarity_matrix.topk(k, dim=1)
    img_to_tab_correct = torch.any(img_to_tab_topk_idx == labels.unsqueeze(1), dim=1)
    img_to_tab_acc = img_to_tab_correct.float().mean()
    
    _, tab_to_img_topk_idx = similarity_matrix.T.topk(k, dim=1)
    tab_to_img_correct = torch.any(tab_to_img_topk_idx == labels.unsqueeze(1), dim=1)
    tab_to_img_acc = tab_to_img_correct.float().mean()

    result = f'K={k}, img2tab: {img_to_tab_acc.item():.3f}, tab2img: {tab_to_img_acc.item():.3f}'
    
    return result


def exists(val):
    return val is not None

    