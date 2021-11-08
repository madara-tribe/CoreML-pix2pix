import sys, os, time
sys.path.append('../')
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import coremltools as ct
from cfg import Cfg
from models.pix2pix import *

cfg = Cfg
transform = [
      transforms.Resize((cfg.img_height, cfg.img_width), Image.BICUBIC),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

def cpu_load_model(model_path):
    # Initialize generator and discriminator
    generator = GeneratorUNet()
    discriminator = Discriminator()

    # Load pretrained models
    generator.load_state_dict(torch.load(os.path.join(model_path, "generator.pth"), map_location=torch.device('cpu')))
    discriminator.load_state_dict(torch.load(os.path.join(model_path, "discriminator.pth"), map_location=torch.device('cpu')))
    return generator, discriminator

Tensor = torch.FloatTensor

def load_img(Ais):
    img = Image.open("cityscapes.jpg")
    w, h = img.size
    img_A = img.crop((0, 0, w / 2, h))
    img_B = img.crop((w / 2, 0, w, h))
    preprocess = transforms.Compose(transform)
    if Ais:
        img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB") 
        input_tensorA = preprocess(img_A)
        input_batch = input_tensorA.unsqueeze(0)
    else:
        img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
        input_tensorB = preprocess(img_B)
        input_batch = input_tensorB.unsqueeze(0)
    return Variable(input_batch.type(Tensor))

def coreml_inference(cuda, model_path, coreml_filename, input_data):
    generator, _ = cpu_load_model(model_path)
    torch_model = generator
    
    print('start calculation')
    start = time.time()
    with torch.no_grad():
        torch_output = {"output": torch_model(input_data)}
    coreml_model = ct.models.MLModel(coreml_filename, useCPUOnly=True)
    # convert input to numpy and get coreml model prediction
    pred = coreml_model.predict({"input": input_data.cpu().numpy()})
    print(pred['var_293'].shape, pred['var_293'].max(), pred['var_293'].min())
    predict_time = time.time() - start
    print("Inference Latency (milliseconds) is", predict_time*1000, "[ms]")
    
    
    key_name='var_293'
    print("CoreML model is checked!")
    pred = pred[key_name] * 127.5 + 127.5
    pred = np.squeeze(pred.transpose(0, 2, 3, 1), axis=0)
    print(pred.shape, pred.max(), pred.min())
    cv2.imwrite('coreml_output.png', pred.astype(np.uint8))

if __name__=='__main__':
    cuda = False #True if torch.cuda.is_available() else False
    path='../tb'
    coreml_path='pix2pix.mlmodel'
    input_batch = load_img(Ais=False)
    save_image(input_batch, 'input_batch.png', nrow=5, normalize=True)
    coreml_inference(cuda, path, coreml_path, input_batch)    

