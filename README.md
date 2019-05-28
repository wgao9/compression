# Model compression

This is the code for the paper [*Rate Distortion For Model Compression: From Theory To Practice*](http://proceedings.mlr.press/v97/gao19c/gao19c.pdf) published on ICML 2019.

To use this code, it is required to install **Python3**, with latest *NumPy*, *PyTorch*, *TorchVision*.


Two steps for running experiments:


**Step 1**: compute weight importances by running
```
python3 get_importance.py --type=mnist --mode=normal
```
Then the weight importances will be computed and stored in weight_importances/mnist.


**Step 2**: load weight importances and run
```
python3 pruning.py --type=mnist --mode=normal --fix_ratio=0.1
```
for weight pruning and
```
python3 quantization.py --type=mnist --mode=normal --fix_bit=4
```
for weight quantizaiton.

**Important arguments**

1. *--type* denotes the dataset and model architecture. Supported: mnist, cifar10, cifar100

2. *--mode* denotes the objective for model compression. Supported:
    - normal ("Baseline" in Table 1 and 2)
    - KL ("KL" in Table 1)
    - gradient ("gradient" in Table 2)
    - hessian ("hessian" in Table 2)
    - For "gradient+hessian" in Table 2, use --mode=gradient and --ha=0.5
    
3. *--fix_ratio* denotes the compression ratio for model pruning (same for all layer). If you want different ratio for different layer, use --ratio=[0.1,0.2,0.3] for example.

4. *--fix_bits* denotes the bits for model compression (same for all layer). If you want different ratio for different layer, use --bits=[3,4,5] for example

5. Please see the helps in the code for other arguments.

**Contact** wgao9@illinois.edu
