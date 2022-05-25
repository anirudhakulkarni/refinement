import os

import torch
from utils import parse_args

from solvers.runners import test
from solvers.loss import loss_dict
from time import localtime, strftime

from models import model_dict
from datasets import dataloader_dict, dataset_nclasses_dict, dataset_classname_dict
import numpy as np

import logging

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
    model.load_state_dict(saved_model_dict['state_dict'])
    model.cuda()

    # set up dataset
    logging.info(f"Using dataset : {args.dataset}")
    trainloader, valloader, testloader = dataloader_dict[args.dataset](args)
    
    criterion = loss_dict[args.loss](gamma=args.gamma, alpha=args.alpha, beta=args.beta, loss=args.loss)
    test_loss, top1, top3, top5, sce_score, ece_score,auroc = test(testloader, model, criterion)
    logging.info("Stats: test_loss : {:.4f} | top1 : {:.4f} | top3 : {:.4f} | top5 : {:.4f} | SCE : {:.5f} | ECE : {:.5f} | AUROC : {:5f}".format(
        test_loss,
        top1,
        top3,
        top5,
        sce_score,
        ece_score,
        auroc['auc']
    ))    
    # save the tpr and fpr
    auroc_name=args.model+'_'+args.dataset+'_'+args.loss+'_'+strftime("%d-%b", localtime())+"_tpr_fpr.npy"
    np.save(os.path.join(args.aurocfolder, auroc_name), auroc['fpr'].append(auroc['tpr']))
