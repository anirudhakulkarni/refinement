import os
from utils.misc import AverageMeter

import torch
import torch.optim as optim
import torch.nn as nn

from utils import Logger, parse_args
from utils import mkdir_p, parse_args,get_trained_loss, create_save_path
from time import localtime, strftime

from solvers.runners import test

from models import model_dict
from datasets import corrupted_dataloader_dict, dataset_nclasses_dict, dataset_classname_dict, corrupted_dataset_dict

import logging
import json
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
    model = nn.DataParallel(model)
    model.load_state_dict(saved_model_dict['state_dict'])
    model.cuda()

    # set up dataset
    logging.info(f"Using dataset : {args.dataset}")

    # set up metrics
    # ece_evaluator = ECELoss(n_classes=num_classes)    
    # fastcce_evaluator = CCELossFast(n_classes=num_classes)
    
    criterion = torch.nn.CrossEntropyLoss()

    # set up loggers
    metric_log_path = os.path.join(checkpoint_folder, 'ood_test.txt')
    logger = Logger(metric_log_path, resume=os.path.exists(metric_log_path))

    logger.set_names(['method', 'test_nll', 'top1', 'top3', 'top5', 'SCE', 'ECE','AUROC'])
    # read corruptions
    corruption_list = ["art", "cartoon", "sketch"]
    
    top1_avg = AverageMeter()
    top3_avg = AverageMeter()
    sce_avg = AverageMeter()
    ece_avg = AverageMeter()
    test_nll_avg = AverageMeter()
    auroc_avg = AverageMeter()
    for c_type in corruption_list:
        _, _, testloader = corrupted_dataloader_dict[args.dataset](args, target_type=c_type)
        test_loss, top1, top3, top5, cce_score, ece_score,auroc = test(testloader, model, criterion)
        method_name = c_type
        logger.append([method_name, test_loss, top1, top3, top5, cce_score, ece_score, auroc["auc"]])

        top1_avg.update(top1)
        top3_avg.update(top3)
        sce_avg.update(cce_score)
        ece_avg.update(ece_score)
        test_nll_avg.update(test_loss)
        auroc_avg.update(auroc["auc"])
    logger.append(["avg_domains", test_nll_avg.avg, top1_avg.avg, top3_avg.avg, top3_avg.avg, sce_avg.avg, ece_avg.avg,auroc_avg.avg])
    # create file with json list
    trained_loss=get_trained_loss(args.checkpoint)

    username=os.getlogin()
    # append test_loss, top1, top3, top5, sce_score, ece_score, auroc as json object
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
        "loss": trained_loss,
        "date": strftime("%d-%b", localtime()),
        "test_loss": test_loss,
        "top1": top1_avg.avg,
        "top3": top3_avg.avg,
        "sce_score": sce_avg.avg,
        "ece_score": ece_avg.avg,
        "auroc": auroc_avg.avg
    })
    with open(jsonfile, 'w') as f:
        json.dump(data, f, indent=4)
    logging.info("Saved results to {}".format(jsonfile))

