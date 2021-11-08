import sys
sys.path.append('../')
import torch
import coremltools as ct
from solver import load_model


def torch2coreml(cuda, model_path, output_name):
    if cuda:
        generator, discriminator = load_model(cuda, model_path)
        dummy_input = torch.randn(1, 3, 256, 256).cuda()
    else:
        generator, discriminator = load_model(cuda, model_path)
        dummy_input = torch.randn(1, 3, 256, 256)
    torchscript_model = torch.jit.trace(generator, dummy_input) 
    mlmodel = ct.convert(torchscript_model, 
                inputs=[ct.TensorType(name="input", shape=dummy_input.shape)])
    mlmodel.save(output_name)
    print(mlmodel)
    print('success convert to ', output_name)

if __name__=='__main__':
    cuda = False
    path='../tb'
    output_name = 'pix2pix.mlmodel'
    torch2coreml(cuda, model_path=path, output_name=output_name)
