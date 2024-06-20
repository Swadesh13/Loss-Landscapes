# Loss-Landscapes
Loss Landscapes of models w/o skips &amp; batchnorm


Credits-

1. https://github.com/tomgoldstein/loss-landscape
2. https://github.com/alxndrTL/losscape

### Install
```
$ git clone https://github.com/Swadesh13/Loss-Landscapes.git
$ cd Loss-Landscapes
$ pip install -r requirements.txt
```

### Train a Model
Simple train ResNet-18 on CIFAR-10 (check arguments)
```
$ python train.py 
```

### Generate Loss Landscape
```
$ python generate_landscapes.py <ckpt-path> --save --output_vtp
```