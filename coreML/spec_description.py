import numpy as np
import coremltools as ct


def coreml_description(coreml_filename):
    coreml_model = ct.models.MLModel(coreml_filename, useCPUOnly=True)
    spec = coreml_model.get_spec()
    print(spec.description)
    
    
if __name__=='__main__':
    coreml_path='pix2pix.mlmodel'
    coreml_description(coreml_path)

