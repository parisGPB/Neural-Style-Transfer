
# Results obtained 

This document has been generated to explain the different results that have been obtained from the Neural Style Transfer repository. All the results have been developed using the Google Cloud platform. A Virtual Machine with 16Gb of RAM and an specific GPU helped in executing all the algorithm iterations. The document will be scheduled as it is shown below:

1. Google Cloud Environment  
2. Results Generation
3. Results Comparison
    * Init variables
    * Iteration results
    * A
    * A
    
    
##  Google Cloud Environment  

The following requirements have been installed in a VM on Google Cloud:

- Theano / Tensorflow
- Keras
- CUDA (GPU) 
- CUDNN (GPU) 
- Scipy + PIL
- Numpy
- h5py

All the requirements have been installed in a virtual environtment with the main repository cloned in it. In the Neural-Style-Transfer folder, a new directory have been declared to save the results of the execution for each iteration.

## Results generation

The main execution code called Network.py has been modified in order to save and manage the information of each iteration. This new Python file store the values of loss, time and improved performance in three independent vectors. At the end of all iterations, this vectors contain the data that is saved in three different graph plots. 

Furthermore, to have a reliable reference of the NST performance, the first 10 iterations are saved (where the improvement is higher) and later on only the multiples of 10. At the end, we obtain a final output directory with 19 images of the style transfer evolution, three more images that contain the data stored (time, loss function value and improvement) and a .txt file with all the parameters used. 

To speed up the process of generating different outputs for each parameter changed, two scripts have been declared. The first one compress the /outputs/ directory to be downloaded, the second one cleans all the directories and prepare the repository to perform another simulation.

## Results comparison

The Neural-Style-Transfer repository have a default configuration of values such as convolutional layers, style weight, content weight or init variables. In the following table all the default parameters are shown. 

| Parameter           | Default Value | Possible values                                                                                                                                      |
|---------------------|---------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| num_iter            | 100           | 200                                                                                                                                                  |
| content_weight      | 0.25          | 0.25 - 1 - 100 - 1                                                                                                                                   |
| style_weight        | 1             | 1 - 0.25 - 1 - 100                                                                                                                                   |
| tv_weight           | 8.5e-5        | 1e-4, 1e-5, 1e-8,  5e-8,1e-20                     |
| region_style_weight | 1             | 1 -0.5 - 0.1- 0.05 - 0.01                                                                                                                            |
| img_size            | 224 x 224     | 224x224- 112x112 - 448 x 448                                                                                                                         |
| --content_loss_type | 0             | 0 - 1 - 2                                                                                                                                            |
| --content_layer     | conv5_2       | conv1_1 - conv1_2 - conv2_1 - conv2_2 - conv3_1 - conv3_2 - conv3_3  - conv4_1 - conv4_2 - conv4_3 - conv4_4 - conv5_1 - conv5_2 - conv5_3 - conv5_4 |
| pool_type           | max           | max  - ave                                                                                                                                           |
| init_image          | noise         | content - noise - gray       


All of this parameters, with the exeption of --content_layer, that has to be modified directly in the code, are set in the initial command when the execution is performed. To have a clear reference of how the Neural Style Transfer is working, 36 different experiments have been proposed. 

The following table contain all the tests made with an identifier. All of this experiments can be found in /2018-dlai-team5/Results/ in which the name of the folder is the identifier of the table.

| Test number | Modified parameter with respect to default              |
|-------------|---------------------------------------------------------|
| 1           | Default configuration                                   |
| 2           | style_weight = 0.25, content_weight = 1                 |
| 3           | style_weight = 100, content_weight = 1                  |
| 4           | style_weight = 1, content_weight = 100                  |
| 5           | tv_weight = 1e-4                                        |
| 6           | tv_weight = 5e-4                                        |
| 7           | tv_weight = 1e-8                                        |
| 8           | tv_weight = 1e-20                                       |
| 9           | ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'] |
| 10          | ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1']            |
| 11          | ['conv1_1', 'conv2_1', 'conv3_1'']                      |
| 12          | ['conv1_1', 'conv2_1']                                  |
| 13          | ['conv1_1']                                             |
| 14          | [ 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']           |
| 15          | ['conv3_1', 'conv4_1', 'conv5_1']                       |
| 16          | [ 'conv4_1', 'conv5_1']                                 |
| 17          | ['conv5_1']                                             |
| 18          | ['conv1_1', 'conv3_1', 'conv5_1']                       |
| 19          | ['conv1_1', 'conv3_1']                                  |
| 20          | Cristian face                                           |
| 21          | Clara face                                              |
| 22          | Guillem face                                            |
| 23          | Marc face                                               |
| 24          | pool_type = ave                                         |
| 25          | init_image = gray                                       |
| 26          | ['conv1_1', 'conv2_1']                                  |
| 27          | Cristian init_image=content                             |
| 28          | Image size 112                                          |
| 29          | Image size 448                                          |
| 30          | Conv layer 1_2                                          |
| 31          | Conv layer 2_2                                          |
| 32          | Conv layer 3_2                                          |
| 33          | Conv layer 5_2                                          |
| 34          | Conv layer 5_2  content_weight = 100 style_weight = 1   |
| 35          | Init image content                                      |

To execute the different tests, two images have been selected as default.

- Content:

![](https://github.com/telecombcn-dl/2018-dlai-team5/blob/master/Utils/Perrete.png)


-Style:

![](https://github.com/telecombcn-dl/2018-dlai-team5/blob/master/Utils/VanGogh.png)


This image was selected because is one of the most typical in datasets of transfer stlyle. The image has a very particular texture and colors, and some elements (like the building and the moon) that are quite characteristic. 


### Results of iterations

As it has been introduced before, in each experiment the Neural Style Transfer code gives us an output image accordint to each iteration. Our modified code provide us he first 10 images according to their iterations and later on the corresponding 9 ones in steps of 10. The following image is an example of all output images in a default parameters iteration:

![](https://github.com/telecombcn-dl/2018-dlai-team5/blob/master/Results/iterations.PNG)
