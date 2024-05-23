# Loss-Landscapes
Loss Landscapes of models w/o skips &amp; batchnorm


Credits-

1. https://github.com/alxndrTL/losscape
2. https://towardsdatascience.com/visualizing-loss-landscape-of-deep-neural-networks-but-can-we-trust-them-3d3ae0cff46e
3. https://github.com/tomgoldstein/loss-landscape

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
$ python generate_landscapes.py <ckpt-path> --load_once --xmin -0.2 --xmax 0.2 --ymin -0.2 --ymax 0.2 --save --output_vtp
```