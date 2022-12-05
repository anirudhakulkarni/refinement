import sys

# import cv2
import numpy as np
import pandas as pd
import torchvision
from PIL import Image
from torchvision import transforms

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
torch.cuda.set_device(0)

###################### loading data ######################

def load_and_resize_img(path):
    """
    Load and convert the full resolution images on CodaLab to
    low resolution used in the small dataset.
    """
    img = cv2.imread(path, 0)

    size = img.shape
    max_dim = max(size)
    max_ind = size.index(max_dim)

    if max_ind == 1:
        # width fixed at 320
        wpercent = (320 / float(size[0]))
        hsize = int((size[1] * wpercent))
        new_size = (hsize, 320)

    else:
        # height fixed at 320
        hpercent = (320 / float(size[1]))
        wsize = int((size[0] * hpercent))
        new_size = (320, wsize)

    resized_img = cv2.resize(img, new_size)

    return resized_img


def ImageTensor(image_name):
    image = Image.open(image_name).convert('RGB')

    transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    image = transform(image)

    return image_name, image



from tkinter import image_names
from libauc.models import densenet121 as DenseNet121

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import sys
import os

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in %r: %s" % (cwd, files))


# paramaters
SEED = 123
BATCH_SIZE = 1
lr = 0.1
gamma = 500
weight_decay = 1e-5
margin = 1.0
epochs = 35


def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
import cv2

def ImageTensor(image_name):
    # image = cv2.imread(image_name, 0)
    image=load_and_resize_img(image_name)
    # print(image,image.shape)
    image = Image.fromarray(image)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # resize and normalize; e.g., ToTensor()
    image = cv2.resize(image, dsize=(320, 320), interpolation=cv2.INTER_LINEAR)  
    image = image/255.0
    __mean__ = np.array([[[0.485, 0.456, 0.406]]])
    __std__ =  np.array([[[0.229, 0.224, 0.225]  ]]) 
    image = (image-__mean__)/__std__
    image = image.transpose((2, 0, 1)).astype(np.float32)
    return image_name, image
# python src/<path-to-prediction-program> <input-data-csv-filename> <output-prediction-csv-path>
input_csv_filename = sys.argv[1]
prediction_csv_filename = sys.argv[2]

df = pd.read_csv(input_csv_filename)
image_list = np.asarray(df)
print(image_list)
# root='../../data/CheXpert-v1.0-small'
try:
    root='./'
    image_list = [root + i for i in image_list]
    print(image_list)
    loader = [ImageTensor(image_name[0]) for image_name in image_list]
except:
    # convert ./CheXpert-v1.0/valid/patient00000/study1/view1_frontal.jpg to valid/patient00000/study1/view1_frontal.jpg
    image_list = [i[0][16:] for i in image_list]
    print(image_list)
    loader = [ImageTensor(image_name) for image_name in image_list]
def test(path):
    # load the saved model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DenseNet121(last_activation=None,
                        activations='relu', num_classes=5)
    model = torch.nn.DataParallel(model, device_ids=[0])
    # model = model.to(device)
    model.load_state_dict(torch.load(
        path,map_location='cuda:0'))
        # 'aucm_multi_label_MDCA_pretrained_model.pth'))
    # model = torch.load('aucm_multi_label_pretrained_model.pth')
    # model = model.cuda()
    # test the model
    model.eval()
    # model = model.to(device)
    pred = []
    y_pred = []
    with torch.no_grad():
        for name, data in loader:
            pred_ = []
            # add a batch dimension
            # numpy.ndarray
            data = np.expand_dims(data, axis=0)
            print(data.shape)
            # data = data.unsqueeze(0)
            data = torch.from_numpy(data)
            data = data.to(device)

            out = model(data)
            out = torch.sigmoid(out).cpu().numpy()
            # print(out)
            pred_.append(name)
            for prob in out[0]:
                pred_.append(prob)

            pred.append(pred_)

        return pred
path='src/src/aucm_multi_label_MDCA_pretrained_model.pth'
try:
    pred=test(path)
except:
    path='aucm_multi_label_MDCA_pretrained_model.pth'
    pred=test(path)
class_names = ['Study', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

pred_df = pd.DataFrame(pred, columns=class_names)
images_names=pred_df['Study']
temp = []
for image_name in images_names:
    image_name = '/'.join(image_name.split('/')[:-1])
    temp.append(image_name)
pred_df['Study'] = temp

def merge_studies(df):
    dic = {'Atelectasis': 'max', 'Cardiomegaly': 'max', 'Consolidation': 'max', 'Edema': 'max', 'Pleural Effusion': 'max'}
    df_new = df.groupby(df['Study'], as_index=False).aggregate(dic).reindex(columns=df.columns)

    return df_new

pred_df = merge_studies(pred_df)

pd.DataFrame.to_csv(pred_df, prediction_csv_filename, index=False)
