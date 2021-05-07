import argparse
import os
import logging
import numpy as np
import pandas as pd
import random
import sys
import io
import torch
from operator import itemgetter
from functools import reduce
import torch.nn.functional as F
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from torch.serialization import default_restore_location
from sklearn.manifold import TSNE   

def add_logging_arguments(parser):
    parser.add_argument("--seed", default=0, type=int, help="random number generator seed")
    parser.add_argument("--output-dir", default="experiments", help="path to experiment directories")
    parser.add_argument("--experiment", default=None, help="experiment name to be used with Tensorboard")
    parser.add_argument("--resume-training", action="store_true", help="whether to resume training")
    parser.add_argument("--restore-file", default=None, help="filename to load checkpoint")
    parser.add_argument("--valid-interval", type=int, default=1, help="validate every N epochs")
    parser.add_argument("--no-save", action="store_true", help="don't save models or checkpoints")
    parser.add_argument("--save-interval", type=int, default=1, help="save a checkpoint every N steps")
    parser.add_argument("--step-checkpoints", action="store_true", help="store all step checkpoints")
    parser.add_argument("--no-log", action="store_true", help="don't save logs to file or Tensorboard directory")
    parser.add_argument("--log-interval", type=int, default=100, help="log every N steps")
    parser.add_argument("--no-visual", action="store_true", help="don't use Tensorboard")
    parser.add_argument("--visual-interval", type=int, default=100, help="log every N steps")
    parser.add_argument("--no-progress", action="store_true", help="don't use progress bar")
    parser.add_argument("--draft", action="store_true", help="save experiment results to draft directory")
    parser.add_argument("--dry-run", action="store_true", help="no log, no save, no visualization")
    return parser


def setup_experiment(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.dry_run:
        args.no_save = args.no_log = args.no_visual = True
        return

    args.experiment = args.experiment or f"{args.model.replace('_', '-')}"
    if not args.resume_training:
        args.experiment = "-".join([args.experiment])

    args.experiment_dir = os.path.join(args.output_dir, args.dataset, (f"drafts/" if args.draft else "") + args.experiment)
    os.makedirs(args.experiment_dir, exist_ok=True)

    if not args.no_save:
        args.checkpoint_dir = os.path.join(args.experiment_dir, "checkpoints")
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    if not args.no_log:
        args.log_dir = os.path.join(args.experiment_dir, "logs")
        os.makedirs(args.log_dir, exist_ok=True)
        args.log_file = os.path.join(args.log_dir, "train.log")


def init_logging(args):
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    handlers = [stream_handler]
    if not args.no_log and args.log_file is not None:
        mode = "a" if os.path.isfile(args.resume_training) else "w"
        file_handler = logging.FileHandler(args.log_file, mode=mode)
        file_handler.setLevel(logging.DEBUG)
        handlers.append(file_handler)
    logging.basicConfig(handlers=handlers, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG)
    logging.info("COMMAND: %s" % " ".join(sys.argv))
    logging.info("Arguments: {}".format(vars(args)))


def save_checkpoint(args, step, model, optimizer=None, scheduler=None, score=None, mode="min"):
    assert mode == "min" or mode == "max"
    last_step = getattr(save_checkpoint, "last_step", -1)
    save_checkpoint.last_step = max(last_step, step)

    default_score = float("inf") if mode == "min" else float("-inf")
    best_score = getattr(save_checkpoint, "best_score", default_score)
    if (score < best_score and mode == "min") or (score > best_score and mode == "max"):
        save_checkpoint.best_step = step
        save_checkpoint.best_score = score

    if not args.no_save and step % args.save_interval == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        model = [model] if model is not None and not isinstance(model, list) else model
        optimizer = [optimizer] if optimizer is not None and not isinstance(optimizer, list) else optimizer
        scheduler = [scheduler] if scheduler is not None and not isinstance(scheduler, list) else scheduler
        state_dict = {
            "step": step,
            "score": score,
            "last_step": save_checkpoint.last_step,
            "best_step": save_checkpoint.best_step,
            "best_score": getattr(save_checkpoint, "best_score", None),
            "model": [m.state_dict() for m in model] if model is not None else None,
            "optimizer": [o.state_dict() for o in optimizer] if optimizer is not None else None,
            "scheduler": [s.state_dict() for s in scheduler] if scheduler is not None else None,
            "args": argparse.Namespace(**{k: v for k, v in vars(args).items() if not callable(v)}),
        }

        if args.step_checkpoints:
            torch.save(state_dict, os.path.join(args.checkpoint_dir, "checkpoint{}.pt".format(step)))
        if (score < best_score and mode == "min") or (score > best_score and mode == "max"):
            torch.save(state_dict, os.path.join(args.checkpoint_dir, "checkpoint_best.pt"))
        if step > last_step:
            torch.save(state_dict, os.path.join(args.checkpoint_dir, "checkpoint_last.pt"))


def load_checkpoint(args, model=None, optimizer=None, scheduler=None):
    if args.restore_file is not None and os.path.isfile(args.restore_file):
        state_dict = torch.load(args.restore_file, map_location=lambda s, l: default_restore_location(s, "cpu"))

        model = [model] if model is not None and not isinstance(model, list) else model
        optimizer = [optimizer] if optimizer is not None and not isinstance(optimizer, list) else optimizer
        scheduler = [scheduler] if scheduler is not None and not isinstance(scheduler, list) else scheduler

        if "best_score" in state_dict:
            save_checkpoint.best_score = state_dict["best_score"]
        if "last_step" in state_dict:
            save_checkpoint.last_step = state_dict["last_step"]
        if model is not None and state_dict.get("model", None) is not None:
            for m, state in zip(model, state_dict["model"]):
                m.load_state_dict(state)
        if optimizer is not None and state_dict.get("optimizer", None) is not None:
            for o, state in zip(optimizer, state_dict["optimizer"]):
                o.load_state_dict(state)
        if scheduler is not None and state_dict.get("scheduler", None) is not None:
            for s, state in zip(scheduler, state_dict["scheduler"]):
                s.load_state_dict(state)

        logging.info("Loaded checkpoint {}".format(args.restore_file))
        return state_dict


def logitexp(logp):
    # Convert outputs of logsigmoid to logits (see https://github.com/pytorch/pytorch/issues/4007)
    pos = torch.clamp(logp, min=-0.69314718056)
    neg = torch.clamp(logp, max=-0.69314718056)
    neg_val = neg - torch.log(1 - torch.exp(neg))
    pos_val = -torch.log(torch.clamp(torch.expm1(-pos), min=1e-20))
    return pos_val + neg_val

def bool_mask(target, context, float=False):
    num_targets, num_contexts = len(target), len(context)
    mask = torch.eq(target.unsqueeze(-1).expand([num_targets, num_contexts]), context.unsqueeze(0).expand([num_targets, num_contexts]))
    if float:
        return mask.float()  # 1 if context label == target label
    else:
        return mask # True if context label == target label

def connectivity(mask, graph, target_labels):
    """
    target_labels: [Torch.Tensor], size: num_targets
    graph: [Torch.Tensor], size: num_targets, num_contexts
    returns [Torch.Tensor], size: num_targets 
    """
    label_connectivity = {}
    pdist = nn.PairwiseDistance(p=2.0)
    gc = pdist(F.normalize(mask,p=1,dim=1), F.normalize(graph,p=1,dim=1))
    for i in torch.unique(target_labels):
        indices = (target_labels==i).nonzero() # list of indices that is i-th label
        scores = [i for i in itemgetter(*indices)(gc)]
        scores = torch.stack(scores)
        label_connectivity[i.item()] = reduce(lambda x,y: x+y, scores).item()/len(scores)
    return label_connectivity

def plot_svd(embedding, task_id, **kwargs):
    u, s, v = torch.svd(embedding)
    s = s.cpu().detach().numpy()
    c = 1-s/np.sum(s, dtype=float)
    figure = plt.figure(figsize=(10,5))
    sns.barplot(x=list(range(1,len(c)+1)),
                y=c[:len(c)])
    plt.title(f'task{task_id}', fontsize=16)
    plt.ylim([0,1.3])
    plt.ylabel('Proportion of Eigenvalue', fontsize=16)

    if "arg" in kwargs:
        args = kwargs.get("arg")
        plt.savefig(os.path.join(args.log_dir,f'svd_eigen_task{task_id}.png'),dpi=100)
    else:
        plt.savefig(f'svd_eigen_task{task_id}.png',dpi=100)
        
    return plt
    
def feature_analysis(embedding, labels,  task_id, columns=["x0","x1","x2"], title="TSNE", **kwargs):
    from sklearn.manifold import TSNE
    from mpl_toolkits.mplot3d import Axes3D
    tsne3 = TSNE(n_components=3, verbose=1,random_state=42, 
                perplexity=40, n_iter=300)
    embedding = embedding.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    X_tsne = tsne3.fit_transform(embedding)
    
    df = pd.DataFrame(X_tsne, columns=columns)
    df['label'] = labels
    cmap = plt.get_cmap('gist_rainbow')
    unique_labels = list(set(labels))
    labels_dict = {label: i for i, label in enumerate(unique_labels)}
    num_colors = len(unique_labels)
    c = [cmap(1.*labels_dict[label]/num_colors) for label in labels]

    df['c'] = c
    df = df[~df['label'].isna()]
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    
    for label, grp in df.groupby('label'):
        ax.scatter(
        xs=grp[columns[0]], 
        ys=grp[columns[1]], 
        zs=grp[columns[2]], 
        c=grp['c'], 
        label=label,
    )
    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])
    ax.set_zlabel(columns[2])
    
    plt.legend(loc='best')
    if title:
        plt.title(title)

    if "arg" in kwargs:
        args = kwargs.get("arg")
        plt.savefig(os.path.join(args.log_dir,f'TSNE_final_embedding_Task{task_id}.png'),dpi=100)
    else:
        plt.savefig(f'TSNE_final_embedding_Task{task_id}.png',dpi=100)
    return plt

def sparsity(graph):
    sp = torch.mean((graph==0).float().sum(dim=1)/ graph.size(1))
    return sp.item() * 100

def combine_graphs(g1, g2):
    """
    g1 : target / context
    g2 : context / context
    """
    