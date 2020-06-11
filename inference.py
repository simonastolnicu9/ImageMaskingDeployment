import json
import logging
import os
import torch
import requests
from PIL import Image
from torchvision import transforms
from torchvision import models
import torch.nn as nn
from torch.autograd import Variable
import base64
from u2net import *

logger = logging.getLogger(__name__)

MODEL_NAME = 'basnet_bsi_epoch_9_train_0.012128_val_0.013882.pth'

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def model_fn(model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Loading the model.')
    
    model = U2NETP(3,1)

    with open(os.path.join(model_dir, MODEL_NAME), 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()
    logger.info('Done loading model')
    return model


def input_fn(request_body, content_type='application/json'):
    logger.info('Deserializing the input data.')
    if content_type == 'application/json':
        input_data = json.loads(request_body)
        url = input_data['url']
        logger.info(f'Image url: {url}')
        image_data = Image.open(requests.get(url, stream=True).raw)

        image_transform = transforms.Compose([RescaleT(320), RandomCrop(288), ToTensorLab(flag=0)])

        return image_transform(image_data)
    raise Exception(f'Requested unsupported ContentType in content_type: {content_type}')


def output_fn(prediction_output, accept='application/json'):
    logger.info('Serializing the generated output.')
    
    result = []
    pred = {'prediction': prediction_output}
    logger.info(f'Adding pediction: {pred}')
    result.append(pred)

    if accept == 'application/json':
        return json.dumps(result), accept
    raise Exception(f'Requested unsupported ContentType in Accept: {accept}')


def predict_fn(input_data, model):
    logger.info('Generating prediction based on input parameters.')
    shapes = input_data.shape
    input_data = input_data.type(torch.FloatTensor)

    if torch.cuda.is_available():
        input_data = Variable(input_data.cuda())
    else:
        input_data = Variable(input_data)

    with torch.no_grad():
        model.eval()
        d1,d2,d3,d4,d5,d6,d7 = model(input_data)
        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)
        predict = predict.squeeze()

        #save image
        predict_np = predict.cpu().data.numpy()

        im = Image.fromarray(predict_np*255).convert('RGB')

        imo = im.resize((shapes[1],shapes[0]),resample=Image.BILINEAR)

        pb_np = np.array(imo)

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1,len(bbb)):
            imidx = imidx + "." + bbb[i]
        
        #ps = torch.exp(imidx)
        ps = base64.b64encode(imidx)

    return ps


