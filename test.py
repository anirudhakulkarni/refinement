import os

import torch
from utils import Logger, parse_args

from solvers.runners import test
from solvers.loss import loss_dict

from models import model_dict
from datasets import dataloader_dict, dataset_nclasses_dict, dataset_classname_dict


import logging

if __name__ == "__main__":
    
    args = parse_args()
    logging.basicConfig(level=logging.INFO, 
                        format="%(levelname)s:  %(message)s",
                        handlers=[
                            # logging.FileHandler(filename=os.path.join(model_save_pth, "train.log")),
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
    # criterion = torch.nn.CrossEntropyLoss()

    # set up loggers
    # metric_log_path = os.path.join(checkpoint_folder, 'temperature.txt')
    # logger = Logger(metric_log_path, resume=os.path.exists(metric_log_path))

    # logger.set_names(['temprature', 'SCE', 'ECE'])

    test_loss, top1, top3, top5, sce_score, ece_score,auroc = test(testloader, model, criterion)
    # logger.append(["1.0", cce_score, ece_score])

    # # Set up temperature scaling
    # temperature_model = TemperatureScaling(base_model=model)
    # temperature_model.cuda()

    # logging.info("Running temp scaling:")
    # temperature_model.calibrate(valloader)
    
    # test_loss, top1, top3, top5, cce_score, ece_score = test(testloader, temperature_model, criterion)
    # logger.append(["{:.2f}".format(temperature_model.T), cce_score, ece_score])
    # logger.close()



    # set up loggers
    # logger = Logger(metric_log_path, resume=os.path.exists(metric_log_path))

    # test_loss, top1, top3, top5, sce_score, ece_score = test(testloader, dir_model, criterion)

    logging.info("Stats: test_loss : {:.4f} | top1 : {:.4f} | top3 : {:.4f} | top5 : {:.4f} | SCE : {:.5f} | ECE : {:.5f} | AUROC : {:s}".format(
        test_loss,
        top1,
        top3,
        top5,
        sce_score,
        ece_score,
        "\n".join("{}\t{}".format(k, v) for k, v in auroc.items())
    ))    
