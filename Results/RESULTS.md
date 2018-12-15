
## Results obtained 

This document has been generated to explain the different results that have been obtained from the Neural Style Transfer repository. All the results have been developed using the Google Cloud platform. A Virtual Machine with 16Gb of RAM and an specific GPU helped in executing all the algorithm iterations. The document will be scheduled as shown below:

1. Google Cloud Environment  
2. Results Generation
3. Results Comparison
    * Output X
    * Output X
    * A
    * A
    
    
#  Google Cloud Environment  

The following requirements have been installed in a VM on Google Cloud:

- Theano / Tensorflow
- Keras
- CUDA (GPU) 
- CUDNN (GPU) 
- Scipy + PIL
- Numpy
- h5py

All the requirements have been installed in a virtual environtment with the main repository cloned in it. In the Neural-Style-Transfer folder, a new directory have been declared to save the results of the execution for each iteration.

# Results generation

The main execution code called Network.py has been modified in order to save and manage the information of each iteration. This new Python file store the values of loss, time and improved performance in three independent vectors. At the end of all iterations, this vectors contain the data that is saved in three different graph plots. 

Furthermore, to have a reliable reference of the NST performance, the first 10 iterations are saved (where the improvement is higher) and later on only the multiples of 10. At the end, we obtain a final output directory with 19 images of the style transfer evolution, three more images that contain the data stored (time, loss function value and improvement) and a .txt file with all the parameters used. 







| Test number | Modified parameter with respect to default |
|-------------|:------------------------------------------:|
| 1           |            Default configuration           |
| 2           |   style_weight = 0.25, content_weight = 1  |
| 3           |   style_weight = 100, content_weight = 1   |
| 4           | style_weight = 1, content_weight = 100     |
| 5           | tv_weight = 1e-4                           |
| 6           | tv_weight = 5e-4                           |
| 7           | tv_weight = 1e-8                           |


Referncia a foto:

![](https://github.com/telecombcn-dl/2018-dlai-team5/blob/master/Results/18/_at_iteration_91.png)
