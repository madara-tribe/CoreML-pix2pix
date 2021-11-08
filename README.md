# Version
```
- cuda 10.1
- nvidia driver 430.64
- python 3.7.0
- pytorch　1.7.1+cu101
- torchvision 0.8.2+cu101
- coremltools 5.0
- onnx 1.10.2
- onnxruntime 1.9.0
```


# coreML inference on ios devices (torch => coreML)
- MacBook Air (M1, 2020)
- memory 16 GB
```
cd coreML
python3 coreml-inference.py
>>>
Inference Latency (milliseconds) is 3367.694854736328 [ms]
```


# References
- [PyTorch to CoreML model conversion](https://learnopencv.com/pytorch-to-coreml-model-conversion/)
