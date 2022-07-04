import argparse
import os

import torch
import torch.optim as optim
import torch.nn as nn
from utils import mkdir_p, parse_args
from utils import get_lr, save_checkpoint, create_save_path
from utils import crl_utils
from solvers.runners import train, test, train_CRL, test_CRL
from solvers.loss import loss_dict

from models import model_dict
from datasets import dataloader_dict, dataset_nclasses_dict, dataset_classname_dict

from time import localtime, strftime
import json
import logging
torch.manual_seed(0)

if __name__ == "__main__":

    args = parse_args()

    current_time = strftime("%d-%b", localtime())
    # prepare save path
    username = os.getlogin()
    model_save_pth = f"{args.checkpoint}/{args.dataset}/{current_time}{create_save_path(args)}_{username}"
    checkpoint_dir_name = model_save_pth

    if not os.path.isdir(model_save_pth):
        mkdir_p(model_save_pth)

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s:  %(message)s",
                        handlers=[
                            logging.FileHandler(filename=os.path.join(
                                model_save_pth, "train.log")),
                            logging.StreamHandler()
                        ])
    logging.info(f"Setting up logging folder : {model_save_pth}")

    num_classes = dataset_nclasses_dict[args.dataset]
    classes_name_list = dataset_classname_dict[args.dataset]

    # prepare model
    logging.info(f"Using model : {args.model}")
    model = model_dict[args.model](num_classes=num_classes,args=args)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    model.cuda()

    # set up dataset
    logging.info(f"Using dataset : {args.dataset}")
    trainloader, valloader, testloader = dataloader_dict[args.dataset](args)

    logging.info(f"Setting up optimizer : {args.optimizer}")

    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)

    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay)
    history = crl_utils.History(len(trainloader.dataset))
    criterion = loss_dict[args.loss](gamma=args.gamma, alpha=args.alpha, beta=args.beta,
                                     loss=args.loss, delta=args.delta, history=history, arguments=args)
    test_criterion = loss_dict["cross_entropy"]()
    logging.info(
        f"Step sizes : {args.schedule_steps} | lr-decay-factor : {args.lr_decay_factor}")
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.schedule_steps, gamma=args.lr_decay_factor)

    start_epoch = args.start_epoch

    best_acc = 0.
    best_acc_stats = {"top1": 0.0}
    if("CRL" in args.loss):
        train = train_CRL
        test = test_CRL

    for epoch in range(start_epoch, args.epochs):

        logging.info('Epoch: [%d | %d] LR: %f' %
                     (epoch + 1, args.epochs, get_lr(optimizer)))

        train_loss, top1_train = train(
            trainloader, model, optimizer, criterion)
        val_loss, top1_val, _, _, sce_score_val, ece_score_val, _ = test(
            valloader, model, test_criterion)
        test_loss, top1, top3, top5, sce_score, ece_score, all_metrics = test(
            testloader, model, test_criterion)

        scheduler.step()

        logging.info("End of epoch {} stats: train_loss: {:.4f} | val_loss: {:.4f} | top1_train: {:.4f} | top1: {:.4f} | SCE: {:.5f} | ECE: {:.5f} | AUROC: {:5f} | FPR-AT-95: {:5f} | AUPR-S: {:5f} | AUPR-E: {:5f} | AURC: {:5f} | EAURC: {:5f}".format(
            epoch+1,
            train_loss,
            test_loss,
            top1_train,
            top1,
            sce_score,
            ece_score,
            all_metrics["auroc"],
            all_metrics["fpr-at-95"],
            all_metrics["aupr-success"],
            all_metrics["aupr-error"],
            all_metrics["aurc"],
            all_metrics["eaurc"]
            # "\n".join("{}\t{}".format(k, v) for k, v in auroc.items())

        ))

        # save best accuracy model
        is_best = top1_val > best_acc
        best_acc = max(best_acc, top1_val)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'dataset': args.dataset,
            'model': args.model
        }, is_best, checkpoint=model_save_pth)

        # Update best stats
        if is_best:
            best_acc_stats = {
                "top1": top1,
                "top3": top3,
                "top5": top5,
                "SCE": sce_score,
                "ECE": ece_score,
                "metrics": all_metrics,
                "epoch": epoch
            }
    # save results to train_results.json
    jsonfile = args.trainresultsfile+"_"+username+".json"
    if not os.path.isfile(jsonfile):
        with open(jsonfile, 'w') as f:
            json.dump({}, f)
    data = []
    if os.stat(jsonfile).st_size != 0:
        data = json.load(open(jsonfile))
    data.append({
        "model": args.model,
        "dataset": args.dataset,
        "loss": args.loss+"_"+args.pairing,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "theta": args.theta,
        "scaling": args.scalefactor,
        "total_epochs": args.epochs,
        "scheduler steps": args.schedule_steps,
        "top3": best_acc_stats["top3"],
        "top5": best_acc_stats["top5"],
        "SCE": best_acc_stats["SCE"],
        "ECE": best_acc_stats["ECE"],
        "top1": best_acc_stats["top1"],
        "AUROC": best_acc_stats["metrics"]["auroc"],
        "FPR-AT-95": best_acc_stats["metrics"]["fpr-at-95"],
        "AUPR-S": best_acc_stats["metrics"]["aupr-success"],
        "AUPR-E": best_acc_stats["metrics"]["aupr-error"],
        "AURC": best_acc_stats["metrics"]["aurc"],
        "EAURC": best_acc_stats["metrics"]["eaurc"],
        "bestepoch": best_acc_stats["epoch"],
        "date": strftime("%d-%b", localtime())
    })
    # "loss": args.loss,
    with open(jsonfile, 'w') as f:
        json.dump(data, f, indent=4)

    logging.info("training completed...")
    logging.info("The stats for best trained model on test set are as below:")
    best_acc_stats["tpr"]=None
    best_acc_stats["fpr"]=None
    logging.info(best_acc_stats)
