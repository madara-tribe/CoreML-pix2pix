# Version
```
- cuda 10.1
- nvidia driver 430.64
- python 3.7.0
- pytorch　1.7.1+cu101
- torchvision 0.8.2+cu101
- coremltools 5.0
- onnx 1.10.2
```
# Abstract

this model pix2pix is mainly for iOs device by converting pytorch to coreML, not onnx to coreML.

<img src="https://user-images.githubusercontent.com/48679574/140760077-2d11343c-fc5e-428b-a102-ef791a732439.png" width="400px"><img src="https://user-images.githubusercontent.com/48679574/140760092-e3870226-baec-4890-b7cc-1eb9f6345ee8.png" width="400px">



# CoreML inference on iOS device (torch => coreML)
- MacBook Air (M1, 2020)
- memory 16 GB
```
cd coreML
python3 inference.py
>>>
Inference Latency (milliseconds) is 3367.694854736328 [ms]
```


## results image

<b>input</b> / <b>output</b>

<img src="https://user-images.githubusercontent.com/48679574/140758490-9b9cb84f-b6b8-4d4f-95a6-6af5f78b6fa3.png" width="400px"><img src="https://user-images.githubusercontent.com/48679574/140758521-ebf06219-7bf6-4d9c-8bca-c31c7130e262.png" width="400px">


## loss curve

<img src="https://user-images.githubusercontent.com/48679574/140757902-fed30192-3c0d-48ce-8964-9ecd4bc0d305.png" width="400px"><img src="https://user-images.githubusercontent.com/48679574/140757913-a85438ae-6e11-4b45-93e2-981413b0a186.png" width="400px">



# References to run CoreML as Xcode project 
- [PyTorch to CoreML model conversion](https://learnopencv.com/pytorch-to-coreml-model-conversion/)
- [coreML](https://developer.apple.com/documentation/coreml)
- [MLModelCamera](https://github.com/shu223/MLModelCamera)
- [CoreMLHelpers](https://github.com/hollance/CoreMLHelpers)
