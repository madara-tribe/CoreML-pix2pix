import cv2, os
import glob
import numpy as np
import torch
import torchvision
from cfg import Cfg
from PIL import Image
import torchvision.transforms as transforms
from solver import load_model
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable

Tensor = torch.FloatTensor
cfg = Cfg
transform = [
      transforms.Resize((cfg.img_height, cfg.img_width), Image.BICUBIC),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
preprocess = transforms.Compose(transform)

def load_img(path, Ais=None):
    img = Image.open(path)
    w, h = img.size
    if Ais:
    	img = img.crop((0, 0, w / 2, h))
    else:
    	img = img.crop((w / 2, 0, w, h))
    img_A = Image.fromarray(np.array(img)[:, ::-1, :], "RGB")
    input_batch = preprocess(img)
    return Variable(input_batch.unsqueeze(0).type(Tensor))
    
    
class Predictor():
    def __init__(self, directory, cuda):
        self.directory = directory
        self.cuda = cuda
        self.model_path = 'tb'
        
    def predictGenerator(self, img_path):
        files = sorted(glob.glob(img_path))
        imgs = [load_img(file, Ais=None) for file in files]
        return imgs
        
    def predict(self, path):
        domeinB=[]
        imgs = self.predictGenerator(path)
        generator, _ = load_model(self.cuda, self.model_path)
        for idx, x in enumerate(imgs):
            print(x.shape)
            output = generator(x) #.float())
            output = make_grid(output, nrow=5, normalize=True)
            inputs_ = make_grid(x, nrow=5, normalize=True)
            output_ = torch.cat((inputs_, output), 2)
            save_image(output_, os.path.join(self.directory, 'pred_img_{}.png'.format(idx)), normalize=False)
            domeinB.append(output)
            if idx==16-1:
                break
        
if __name__ == '__main__':
    directory = "results"
    path = 'datasets/cityscapes/train/*.jpg'
    os.makedirs(directory, exist_ok=True)
    cuda = False
    Predictor(directory, cuda).predict(path)

