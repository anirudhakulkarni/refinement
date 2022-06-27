import os

import torch
from utils import mkdir_p, parse_args,get_trained_loss, create_save_path,crl_utils

from solvers.runners import test, test_CRL
from solvers.loss import loss_dict
from time import localtime, strftime

from models import model_dict
from datasets import dataloader_dict, dataset_nclasses_dict, dataset_classname_dict
import numpy as np
import torch.nn as nn 
import logging
import json

np.random.seed(0)

if __name__ == "__main__":
    
    args = parse_args()
    logging.basicConfig(level=logging.INFO, 
                        format="%(levelname)s:  %(message)s",
                        handlers=[
                            logging.StreamHandler()
                        ])

    num_classes = dataset_nclasses_dict[args.dataset]
    classes_name_list = dataset_classname_dict[args.dataset]
    
    # prepare model
    logging.info(f"Using model : {args.model}")
    assert args.checkpoint, "Please provide a trained model file"
    assert os.path.isfile(args.checkpoint)
    logging.info(f'Resuming from saved checkpoint: {args.checkpoint}')
   
    checkpoint_folder = os.path.dirname(args.checkpoint)
    saved_model_dict = torch.load(args.checkpoint)

    model = model_dict[args.model](num_classes=num_classes, alpha=args.alpha)
    
    try:
        model.load_state_dict(saved_model_dict['state_dict'])
    except:
        model=nn.DataParallel(model)
        model.load_state_dict(saved_model_dict['state_dict'])
    model.cuda()
    if("CRL" in args.loss):
        test = test_CRL

    # set up dataset
    logging.info(f"Using dataset : {args.dataset}")
    trainloader, valloader, testloader = dataloader_dict[args.dataset](args)
    
    history = crl_utils.History(len(trainloader.dataset))
    criterion = loss_dict[args.loss](gamma=args.gamma, alpha=args.alpha, beta=args.beta,
                                     loss=args.loss, delta=args.delta, history=history, arguments=args)
    test_loss, top1, top3, top5, sce_score, ece_score,all_metrics = test(testloader, model, criterion)
    logging.info("Stats: test_loss : {:.4f} | top1 : {:.4f} | top3 : {:.4f} | top5 : {:.4f} | SCE: {:.5f} | ECE: {:.5f} | AUROC: {:5f} | FPR-AT-95: {:5f} | AUPR-S: {:5f} | AUPR-E: {:5f} | AURC: {:5f} | EAURC: {:5f}".format(
        test_loss,
        top1,
        top3,
        top5,
        sce_score,
        ece_score,
        all_metrics["auroc"],
        all_metrics["fpr-at-95"],
        all_metrics["aupr-success"],
        all_metrics["aupr-error"],
        all_metrics["aurc"],
        all_metrics["eaurc"]
    ))    
    # save the tpr and fpr
    if not os.path.isdir(args.aurocfolder):
        mkdir_p(args.aurocfolder)

    trained_loss=get_trained_loss(args.checkpoint)
    auroc_name=args.dataset+"_"+args.checkpoint[11:].split("/")[1]
    print(auroc_name)
    # auroc_name=args.model+'_'+args.dataset+'_'+strftime("%d-%b", localtime())+create_save_path(args)+"_tpr_fpr.npy"
    # np.save(os.path.join(args.aurocfolder, auroc_name), np.append(auroc['tpr'],auroc['fpr']))

    username=os.getlogin()
    # append test_loss, top1, top3, top5, sce_score, ece_score as json object
    jsonfile=args.resultsfile+"_"+username+".json"
    if not os.path.isfile(jsonfile):
        with open(jsonfile, 'w') as f:
            json.dump([{}], f)
    data=[]
    if os.stat(jsonfile).st_size != 0:
        data=json.load(open(jsonfile))
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
        "top3": top3,
        "top5": top5,
        "SCE": sce_score,
        "ECE": ece_score,
        "top1": top1,
        "AUROC": all_metrics["auroc"],
        "FPR-AT-95": all_metrics["fpr-at-95"],
        "AUPR-S": all_metrics["aupr-success"],
        "AUPR-E": all_metrics["aupr-error"],
        "AURC": all_metrics["aurc"],
        "EAURC": all_metrics["eaurc"],
        "date": strftime("%d-%b", localtime())
    })
    with open(jsonfile, 'w') as f:
        json.dump(data, f, indent=4)
    logging.info("Saved results to {}".format(jsonfile))
