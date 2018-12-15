
# STYLE TRANSFER - DLAI TEAM #5
This project is carried out by ETSETB students for the Deep Learning for Artificial Intelligence course. 

## Index:
1. Goals
2. What is Style Transfer?
3. Types of style transfer studied
4. Implementation Overview
5. CNN Structure
6. Loss functions
7. Gram Matrix
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

## CNN Structure
 ![](https://github.com/telecombcn-dl/2018-dlai-team5/blob/master/NN.png)
 
In the image-wise NN, convolutional layers and maxpooling are typically used. Usually, pre-trained networks with large datasets --such as VGG16 & VGG19-- are used. These networks are useful since they have been trained to extract features of the input images.

The first layers extract the most detailed features of the input image (pixel-level). The last layers contain the main features such as ears, mouth, etc. The deepest the layer is chosen, the more the style will be used from that input image. Alternatively, if the chosen layer is extracting low-level features, the content will be more important.

To represent the content image, it is used a high layer. High layers in the network capture the high-level content in terms of objects and their arrangement in the input image but do not constrain the exact pixel values of the reconstruction. In contrast, reconstructions from the lower layers simply reproduce the exact pixel values of the original image.

To represent the style image, which is defined as the artistic features such as textures, pattern, brightnes,etc., it is mandatory to employ a feature space originally designed to capture texture information. The first convolution layers of each block are used. Correlations between the different filter responses over the spatial extent of the feature maps are calculated. By including the feature correlations of multiple layers, it is obtained a stationary, multi-scale representation of the input image, which captures its texture information but not the global arrangement.

## Loss functions
The principle of neural style transfer is to define two distance functions, one that describes how different the content of two images are, Lcontent, and one that describes the difference between the two images in terms of their style, Lstyle. Then, given three images, a desired style image, a desired content image, and the input image (initialized with the content image or some noise), we try to transform the input image to minimize the content distance with the content image and its style distance with the style image.
In summary, we’ll take the base input image, a content image that we want to match, and the style image that we want to match. We’ll transform the base input image by minimizing the content and style distances (losses) with backpropagation, creating an image that matches the content of the content image and the style of the style image. (?)

In this case, the loss function will be formed by the content-image loss function --which represents how far is the generated image from the content one-- and the style-image loss function --which represents how well the style has been emulated--. (FORMULASSSS!).

![](https://cdn-images-1.medium.com/max/1600/1*Wd0L4_LA77g5cLWon7L3Hw.png)

![](https://cdn-images-1.medium.com/max/1600/1*3LnRslYfEIqdLmVDP3PP3w.png)

![](https://cdn-images-1.medium.com/max/1600/1*F3yL2YQCQ3BH3cGWBRF9Hw.png)


## Gram Matrix
The Gram Matrix is used to compare both the style image and the output one.
The style representation of an image is described as the correlation of the different filter responses given by the Gram matrix.
Given the first layer of the trained network and a CxHxW vector space is obtained, where C is the number of filters, H is the height of the image and W the width. From these parameters, we compute the Gram Matrix. To obtain it, different rows are chosen and their inner product computed in order to see which neurons tend to be activated at the same time.

(FORMULA G^l_ij)
"where Gˡᵢⱼ is the inner product between the vectorized feature map i and j in layer l. We can see that Gˡᵢⱼ generated over the feature map for a given image represents the correlation between feature maps i and j."

## Basic Neural Style Transfer

### Results
This section will sum up the results we have obtained during the carried tests.

## Improved Neural Style Transfer

### Results
This section will sum up the results we have obtained during the carried tests.

## Fast Neural Style Transfer

## Arbitrary Neural Style Transfer

## References
- Basic Style Transfer & Improved Style Transfer: https://github.com/titu1994/Neural-Style-Transfer
