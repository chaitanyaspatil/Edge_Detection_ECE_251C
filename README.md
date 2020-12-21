## Integrating the Wavelet Transform into the Holistically-Nested Edge Detection (HED) Algorithm

#### Introduction

We combine the [holistically-nested edge detection (HED)](https://arxiv.org/abs/1504.06375) algorithm with the discrete wavelet transform in an attempt to improve the performance of the algorithm. We use an [existing PyTorch implementation](https://github.com/xwjabc/hed) of the official HED code (originally written in Caffe). We have used two different versions of the wavelet transform - Haar and the biorthogonal 4.4 wavelet. We will describe our various approaches to this problem and review the results for the two situations, along with possible explanations for our results. 

Running this code requires access to Python 3.6, PyTorch 1.0 (CUDA9, CUDNN7) and MATLAB R2018b.

#### Instructions

##### Prepare

1. Clone the repository:

   ```bash
   git clone https://github.com/chaitanyaspatil/Edge_Detection_ECE_251C.git
   ```

2. Download and extract the data:

   ```bash
   cd Edge_Detection_ECE_251C
   wget https://cseweb.ucsd.edu/~weijian/static/datasets/hed/hed-data.tar
   tar xvf ./hed-data.tar
   ```
##### Generating the Haar and Bior wavelet-decomposed datasets

1. Generate the Haar dataset: 

For training data: Go to `dataset_generator_final.ipynb` and click Restart and Run All.
For test data: Go to `Test_dataset_generation_haar.ipynb` and click Restart and Run All.

2. Generate the biorthogonal dataset:

For training data: Go to `dataset_generator_final.ipynb` and click Restart and Run All.
For test data: Go to `test_dataset_generator_bior_final.ipynb` and click Restart and Run All.

##### Train, Recombine, and Evaluate

1. Train the 4 CNNs for the Haar wavelet decomposed images:

   ```bash
   python hed_LL.py --vgg16_caffe ./data/5stage-vgg.py36pickle
   python hed_LH.py --vgg16_caffe ./data/5stage-vgg.py36pickle
   python hed_HL.py --vgg16_caffe ./data/5stage-vgg.py36pickle
   python hed_HH.py --vgg16_caffe ./data/5stage-vgg.py36pickle
   ```
   
   By default, the results are in the `output/LL`, `output/LH`, `output/HL`, or `output/HH` folder and the models are trained for 40 epochs. 
   
2. Train the 4 CNNs for the bior(4, 4) wavelet decomposed images:

For the LL network: Go to `bior_debug.ipynb` and click Restart and Run All.
For the LH network: Go to `bior_debug_LH.ipynb` and click Restart and Run All.
For the HL network: Go to `bior_debug_HL.ipynb` and click Restart and Run All.
For the HH network: Go to `bior_debug_HH.ipynb` and click Restart and Run All.
   
   By default, the results are in the `output/HH_bior`, `output/HH_bior`, `output/HH_bior`, or `output/HH_bior` folder and the models are trained for 40 epochs. 

3. Recombine the Haar output images:
Go to `IDWT_of_LL_LH_HL_HH.ipynb` and Restart and Run All

4. Recombine the bior(4, 4) output images:
Go to `IDWT_of_LL_LH_HL_HH.ipynb` and Restart and Run All

5. Evaluate the Haar wavelet decomposed images:

   ```bash
   cd eval
   (echo "data_dir = '../output_LL_haar/epoch-39-test'"; cat eval_edge.m)|matlab -nodisplay -nodesktop -nosplash
   ```
   

6. Evaluate the bior(4, 4) wavelet decomposed images:

   ```bash
   cd eval
   (echo "data_dir = '../output_LL_bior/epoch-4-test'"; cat eval_edge.m)|matlab -nodisplay -nodesktop -nosplash
   ```

   The evaluation process for each set of images takes ~7hrs.

Based on the observations of @xjwabc, the evaluated performance for the networks is somewhat stable after 5 epochs.

##### Evaluate the Pre-trained Models

1. Evaluate [@xjwabc](https://github.com/xwjabc/)'s pre-trained version:

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
