## Integrating the Wavelet Transform into the Holistically-Nested Edge Detection (HED) Algorithm

#### Introduction

We combine the [holistically-nested edge detection (HED)](https://arxiv.org/abs/1504.06375) algorithm with the discrete wavelet transform in an attempt to improve the performance of the algorithm. We use an [existing PyTorch implementation](https://github.com/xwjabc/hed) of the official HED code (originally written in Caffe). We have used two different versions of the wavelet transform - Haar and the biorthogonal 4.4 wavelet. We will describe our various approaches to this problem and review the results for the two situations, along with possible explanations for our results. 

The code is evaluated on Python 3.6 with PyTorch 1.0 (CUDA9, CUDNN7) and MATLAB R2018b.

#### Instructions

##### Prepare

1. Clone the repository:

   ```bash
   git clone https://github.com/xwjabc/hed.git
   ```

2. Download and extract the data:

   ```bash
   cd hed
   wget https://cseweb.ucsd.edu/~weijian/static/datasets/hed/hed-data.tar
   tar xvf ./hed-data.tar
   ```

##### Train and Evaluate

1. Train:

   ```bash
   python hed.py --vgg16_caffe ./data/5stage-vgg.py36pickle
   ```

   The results are in `output` folder. In the default settings, the HED model is trained for 40 epochs, which takes ~27hrs with one NVIDIA Geforce GTX Titan X (Maxwell). 

2. Evaluate:

   ```bash
   cd eval
   (echo "data_dir = '../output/epoch-39-test'"; cat eval_edge.m)|matlab -nodisplay -nodesktop -nosplash
   ```

   The evaluation process takes ~7hrs with Intel Core i7-5930K CPU @ 3.50GHz.

Besides, based on my observation, the evaluated performance is somewhat stable after 5 epochs (5 epochs: **ODS=0.788 OIS=0.808** vs. 40 epochs: **ODS=0.787 OIS=0.807**).

##### Evaluate the Pre-trained Models

1. Evaluate the my pre-trained version:

   ```bash
   python hed.py --checkpoint ./data/hed_checkpoint.pt --output ./output-mypretrain --test
   cd eval
   (echo "data_dir = '../output-mypretrain/test'"; cat eval_edge.m)|matlab -nodisplay -nodesktop -nosplash
   ```

   The result should be similar to **ODS=0.787 OIS=0.807**.

2. Evaluate the official pre-trained version:

   ```bash
   python hed.py --caffe_model ./data/hed_pretrained_bsds.py36pickle --output ./output-officialpretrain --test
   cd eval
   (echo "data_dir = '../output-officialpretrain/test'"; cat eval_edge.m)|matlab -nodisplay -nodesktop -nosplash
   ```

   The result should be similar to **ODS=0.788 OIS=0.806**.

#### Acknowledgement

This project is based on a lot of previous work. Thank you to [@xjwabc](https://github.com/xwjabc/hed) for their reimplementation of the original HED algorithm in PyTorch that we used. This README.md also borrows from their instructions. Thank you also to [Saining Xie](https://github.com/s9xie/hed) for their paper that this project is based on and the original Caffe implementation. 
