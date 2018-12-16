
# STYLE TRANSFER - DLAI TEAM #5
This project is carried out by ETSETB students for the Deep Learning for Artificial Intelligence course. 

## Index:
1. Goals
2. What is Style Transfer?
3. Types of style transfer studied
4. Basic Neural Style Transfer

      * CNN Structure
      * Loss functions
      * Gram Matrix
      * Main hyperparameters
5. Improved Neural Style Transfer
6. Fast Neural Style Transfer
7. Arbitrary Neural Style Transfer
8. Results
9. References

## Goals:
- Understand the basics of Neural Style Transfer (NST)
- Experiment with the different hyperparameters
- Study the different NST techniques: Improved, Fast & Arbitrary Fast Style Transfer

## What is Style Transfer?
Neural style transfer is an optimization technique used to take three images, a content image, a style reference image (such as an artwork by a famous painter), and the input image you want to style — and blend them together such that the input image is transformed to look like the content image, but “painted” in the style of the style image.

## Types of style transfer studied
- Basic Style Transfer
- Improved Style Transfer
- Fast Neural Style Transfer
- Arbitrary Neural Style Transfer

## Basic Neural Style Transfer
### CNN Structure
 ![](https://github.com/telecombcn-dl/2018-dlai-team5/blob/master/NN.png)
In the image-wise NN, convolutional layers and maxpooling are typically used. Usually, pre-trained networks with large datasets --such as VGG16 & VGG19-- are used. These networks are useful since they have been trained to extract features of the input images.

The first layers extract the most detailed features of the input image (pixel-level). The last layers contain the main features such as ears, mouth, etc. The deepest the layer is chosen, the more the style will be used from that input image. Alternatively, if the chosen layer is extracting low-level features, the content will be more important.

To represent the content image, it is used a high layer. High layers in the network capture the high-level content in terms of objects and their arrangement in the input image but do not constrain the exact pixel values of the reconstruction. In contrast, reconstructions from the lower layers simply reproduce the exact pixel values of the original image.

To represent the style image, which is defined as the artistic features such as textures, pattern, brightnes,etc., it is mandatory to employ a feature space originally designed to capture texture information. The first convolution layers of each block are used. Correlations between the different filter responses over the spatial extent of the feature maps are calculated. By including the feature correlations of multiple layers, it is obtained a stationary, multi-scale representation of the input image, which captures its texture information but not the global arrangement.

### Loss functions
The principle of neural style transfer is to define two distance functions, one that describes how different the content of two images are, Lcontent, and one that describes the difference between the two images in terms of their style, Lstyle. Then, given three images, a desired style image, a desired content image, and the input image (initialized with the content image or some noise), we try to transform the input image to minimize the content distance with the content image and its style distance with the style image.
In summary, we’ll take the base input image, a content image that we want to match, and the style image that we want to match. We’ll transform the base input image by minimizing the content and style distances (losses) via backpropagation, creating an image that mixes the content and the style of both images.

In this case, the loss function will be formed by the content-image loss function --which represents how far is the generated image from the content one-- and the style-image loss function --which represents how well the style has been emulated--. Alpha and Beta are the weighting factors for content style reconstruction. (In the formula bewlo, beta is set to 1).

![](https://cdn-images-1.medium.com/max/1600/1*Wd0L4_LA77g5cLWon7L3Hw.png)

![](https://cdn-images-1.medium.com/max/1600/1*3LnRslYfEIqdLmVDP3PP3w.png)

![](https://cdn-images-1.medium.com/max/1600/1*F3yL2YQCQ3BH3cGWBRF9Hw.png)

The layers used habitually for Style Loss Function are: conv1_1, conv2_1, conv3_1, conv4_1, conv5_1.


### Gram Matrix
The Gram Matrix is used to compare both the style image and the output one.
The style representation of an image is described as the correlation of the different filter responses given by the Gram matrix.
Given the first layer of the trained network a CxHxW vector space is obtained, where C is the number of filters, H is the height of the image and W the width. From these parameters, we compute the Gram Matrix. To obtain it, different rows are chosen and their inner product is computed in order to see which neurons tend to be activated at the same time.

(FORMULA G^l_ij)
"where Gˡᵢⱼ is the inner product between the vectorized feature map i and j in layer l. We can see that Gˡᵢⱼ generated over the feature map for a given image represents the correlation between feature maps i and j."

### Main hyperparameters

The follow hyperparameters can be used to modify the output of the transfer style. Differents tests have been done changing the different defauled values and will be discussed in the results section.

- Content weight: weight of the content image in the generated image. Value by default: 0.25.

- Style weight: weight of the style image in the generated image. Value by default 1.

- Total variations weight: Total variation loss is the sum of the absolute differences for neighboring pixel-values in the input images. Smaller values tend to produce sharper images. Value by default: 8,5e-5

- Image size: size of the generated image. Value by default: 224x224x3

- Content layer: Layer from the model used to evaluate similarity between content image and generated image. Value by default: conv4_2

- Feature layers: Layers from the model used to evaluate similarity between style image and generated image. Value by default: conv1_1, conv2_1, conv3_1, 'conv4_1, conv5_1.

- Pool type: use max pooling or average pooling in the model. Value by default: Max pooling

- Initial image: define the initial image that will be modified at each iteration. 

- Preserve color: apply modifications in image style in order to preserve color from the content image. By default this option is off.

## Improved Neural Style Transfer

This is a variation of the basic neural style transfer. The main improvements are:

- Geometric Layer weight adjustment for Style inference

Define the weight of style layers and content layer for each style layer.

![](https://github.com/telecombcn-dl/2018-dlai-team5/blob/master/Utils/formula.png)

Where:
D: is the number of layers.
d(l): is the deeper of the layer l. 
     
- Using all layers of VGG-16 for style inference

Increase the number of layers used to calculate the transfer style loss. 
     
- Activation Shift of gram matrix
     
In general, the outputs of the image are scarce: in all the layers, each filter has few activations different from the    spatial dimensions. This results in the scattering ot the Gram matrices, at the expense of the transfer quality of the style. Gram matrices contain a large number of zero entries, leaving too much freedom for the optimization procedure to interpret them incorrectly.

This problem can be reduced by applying changed activations to eliminate dispersion.

Original Gram Matrix:

![](https://github.com/telecombcn-dl/2018-dlai-team5/blob/master/Utils/formula1.png)

Gram Matrix with Shift activation:

![](https://github.com/telecombcn-dl/2018-dlai-team5/blob/master/Utils/formula2.png)
     
- Correlation Chain

Style information is captured by a set of Gram matrices calculated through correlations within the same layer. 
The correlation chain consists of adding correlations between neighbor layers.

- Color preservation

Luminance channels (LS and LC) are first extracted from the style and content images to produce an output luminance image LT. This transformation is applied before running the Style Transfer algorithm.  Using the YIQ color space, the color information of the content image is represented via the I and Q channels; these are combined with LT to produce the final color output image. 
Being µS and µC the mean luminance of the two images and σS and σC their standard deviations, each luminance pixel in the style image is updated as:

![](https://github.com/telecombcn-dl/2018-dlai-team5/blob/master/Utils/formula4.png)

There are others ways to implement Color Preservation in Style Transfer, but here it is explained the one it has been implemented.

## Fast Neural Style Transfer

To reduce the time needed from the original transfer style.

- Perceptual Losses for Real-Time Style Transfer and Super-Resolution:

Feed-forward networks are trained to solve the optimization problem. 

Results are similar to both in terms of quality (and objective measurement) but ~1000x times faster to generate. However, it is lost the flexibility of the original Style Transfer that can combine two arbitrary images. This network allows to apply only one style transformation per architecture. 
The system consists of two components: the image transformation network and the loss network.

![](https://github.com/telecombcn-dl/2018-dlai-team5/blob/master/Utils/formula5.png)

1. Image transformation network

The image transformation network is a deep residual convolutional neural network. Each block have the follow structure:

![](https://github.com/telecombcn-dl/2018-dlai-team5/blob/master/Utils/formula6.png)

This network comprises five residual blocks. The first and last layers use 9×9 kernels; all other convolutional layers use 3×3 kernels.

2. Loss network

Loss function computes a scalar value measuring the difference between the output image and a target image. The image transformation network is trained using stochastic gradient descent to minimize a weighted combination of loss functions:
Loss function is calculated in the same way than the loss in original Transfer Style.

Instance Normalization: The Missing Ingredient for Fast Stylization

Instance Normalization is used to replace batch normalization. While batch normalization applies the normalization to a whole batch of images, instance Normalization works for a single image separately. The objective is to replace every batch normalization in the generator network with the instance normalization in order to prevent instance-specific mean and covariance shift simplifying the learning process.

Batch Normalization
![](https://github.com/telecombcn-dl/2018-dlai-team5/blob/master/Utils/formula7.png)

Instance Normalization
![](https://github.com/telecombcn-dl/2018-dlai-team5/blob/master/Utils/formula8.png)

Where: 
T: batch size;
W: image weight;
H: image height

## Arbitrary Neural Style Transfer

This solution combine the flexibility of the basic Neural Style Transfer and the speed of the Fast Transfer Style.

Training this new algorithm with a large number of style paintings, provide an algorithm capable to understand new styles never previously observed. 

This model is based in 3 modules: Style prediction network, style transfer network and loss network. This model provides a generic representation for artistic styles that seems flexible enough to capture new artistic styles much faster than a single-purpose network.

- Style transfer network

Reduces each style image into a point in an embedding space employing a pretrained Inception-v3 architecture and compute the mean across of each activation channel of the Mixed-6e layer which returns a feature vector with the dimension of 768. Then apply two fully connected layers on top of it to predict the final embedding S.

- Style prediction network

Calculate the termed conditional instance normalization, which is based on applying a normalization with parameters that are specific for each style transfer. 
![](https://github.com/telecombcn-dl/2018-dlai-team5/blob/master/Utils/formula10.png)

Where 
     * mu is the mean
     * Sigma is the standard deviation
     * Beta and gamma are learned by the neural network. Beta is the learned mean and gamma is the learned standard deviation

The style prediction network is implemented using convolutional layers and residual blocks.

- Loss network

The loss function is calculated applying the structure previously defined and uses the VGG16 architecture. 

![](https://github.com/telecombcn-dl/2018-dlai-team5/blob/master/Utils/formula9.png)




## Results
All the results have been stored in the Results folder and explained in the RESULTS.md document located inside it.

## References
- Basic Style Transfer & Improved Style Transfer: https://github.com/titu1994/Neural-Style-Transfer
- Improved Style Transfer: https://arxiv.org/pdf/1605.04603.pdf
- Color preservation: https://arxiv.org/pdf/1606.05897.pdf
- Perceptual losses for Real-Time Style Transfer and Super-Resolution: https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf
- Instance Normalization: The Missing Ingredient for Fast Stylization:
https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf
- Arbitrary Fast Style Transfer: https://arxiv.org/pdf/1705.06830.pdf , https://github.com/tensorflow/magenta/tree/master/magenta/models/arbitrary_image_stylization

