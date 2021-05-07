import argparse
import logging
import torch
import json
import os
from torch.utils.tensorboard import SummaryWriter

import data, models, optim, utils


def main(args):
    if not torch.cuda.is_available():
        raise NotImplementedError("Training on CPU is not supported.")
    utils.setup_experiment(args)
    utils.init_logging(args)

    train_loaders, valid_loaders = data.build_dataset(args.dataset, args.data_path, batch_size=args.batch_size)
    model = models.build_model(args).cuda()
    optimizer = optim.build_optimizer(args, model.parameters())

    logging.info(f"Built a model consisting of {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    logging.getLogger('matplotlib.font_manager').disabled = True

    meters = {name: utils.RunningAverageMeter(0.98) for name in (["loss", "context", "graph", "target"])}
    acc_names = ["overall"] + [f"task{idx}" for idx in range(len(valid_loaders))] + ["A_sparsity", "label_sparsity"]
    acc_meters = {name: utils.AverageMeter() for name in acc_names}
    writer = SummaryWriter(log_dir=args.experiment_dir) if not args.no_visual else None

    global_step = -1
    for epoch in range(args.num_epochs):
        acc_tasks = {f"task{idx}": None for idx in range(len(valid_loaders))}
        for task_id, train_loader in enumerate(train_loaders):
            for repeat in range(args.num_repeats_per_task):
                train_bar = utils.ProgressBar(train_loader, epoch, prefix=f"task {task_id}")
                for meter in meters.values():
                    meter.reset()

                for batch_id, (images, labels) in enumerate(train_bar):
                    model.train()
                    global_step += 1
                    images, labels = images.cuda(), labels.cuda()
                    outputs = model(images, labels, task_id=task_id)

                    if global_step == 0:
                        continue
                    loss = outputs["loss"]
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()

                    meters["loss"].update(loss.item())
                    meters["context"].update(outputs["context_loss"].item())
                    meters["target"].update(outputs["target_loss"].item())
                    meters["graph"].update(outputs["graph_loss"].item())
                    train_bar.log(dict(**meters, lr=optimizer.get_lr(),))
                        
                if writer is not None:
                    writer.add_scalar("loss/train", loss.item(), global_step)
                    gradients = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None], dim=0)
                    writer.add_histogram("gradients", gradients, global_step)

            model.eval()
            for meter in acc_meters.values():
                meter.reset()
            final_embeddings = []
            final_labels = []
            for idx, valid_loader in enumerate(valid_loaders):
                if idx>task_id:
                    break
                valid_bar = utils.ProgressBar(valid_loader, epoch, prefix=f"task {task_id}")
                for batch_id, (images, labels) in enumerate(valid_bar):
                    model.eval()
                    with torch.no_grad():
                        images, labels = images.cuda(), labels.cuda()
                        outputs = model.predict(images, labels, task_id=idx)
                        correct = outputs["preds"].eq(labels).sum().item()
                        acc_meters[f"task{idx}"].update(100 * correct, n=len(images))
                   
                acc_meters["overall"].update(acc_meters[f"task{idx}"].avg)
                final_embeddings.append(outputs["final_embeddings"])
                final_labels.append(outputs["target_labels"])

            acc_tasks[f"task{task_id}"] = acc_meters[f"task{task_id}"].avg
            if writer is not None:
                for name, meter in acc_meters.items():
                    writer.add_scalar(f"accuracy/{name}", meter.avg, global_step)
            logging.info(train_bar.print(dict(**meters, **acc_meters, lr=optimizer.get_lr())))
            utils.save_checkpoint(args, global_step, model, optimizer, score=acc_meters["overall"].avg, mode="max")

    bwt = sum(acc_meters[task].avg - acc for task, acc in acc_tasks.items()) / (len(valid_loaders) - 1)
    logging.info(f"Done training! Final accuracy {acc_meters['overall'].avg:.4f}, backward transfer {bwt:.4f}.")


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Add data arguments
    parser.add_argument("--data-path", default="data", help="path to data directory")
    parser.add_argument("--dataset", default="split_cifar10", help="train dataset name", choices=["split_cifar10", "split_svhn", "permuted_mnist", "rotated_mnist"])
    parser.add_argument("--batch-size", default=10, type=int, help="train batch size")

    # Add model arguments
    parser.add_argument("--model", default="gcl", help="model architecture")

    # Add optimization arguments
    parser.add_argument("--optimizer", default="adam", help="optimizer")
    parser.add_argument("--lr", default=2e-4, type=float, help="learning rate")
    parser.add_argument("--num-repeats-per-task", default=1, type=int, help="number of repeats per task")
    parser.add_argument("--num-epochs", default=1, type=int, help="force stop training at specified epoch")

    parser.add_argument("--in-channels", default=3, type=int, help="input dimension")
    parser.add_argument("--num-classes", default=10, type=int, help="number of classes")
    parser.add_argument("--buffer-size", default=500, type=int, help="buffer size")
    parser.add_argument("--hidden-size", default=256, type=int, help="hidden dimension")
    parser.add_argument("--context-lambda", default=1, type=float, help="context lambda")
    parser.add_argument("--graph_lambda", default=50, type=float, help="graph lambda")
    parser.add_argument("--context-temperature", default=1.0, type=float, help="temperature")
    parser.add_argument("--target-temperature", default=5.0, type=float, help="temperature")


    # Parse twice as model arguments are not known the first time
    parser = utils.add_logging_arguments(parser)
    args, _ = parser.parse_known_args()
    optim.OPTIMIZER_REGISTRY[args.optimizer].add_args(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
