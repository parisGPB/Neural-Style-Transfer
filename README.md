
# STYLE TRANSFER - DLAI TEAM #5

## Goals:
- Emulate the Style Transfer project
- Understand
- Modify some hyper-parameters (?)

## What is Style Transfer?
Neural style transfer is an optimization technique used to take three images, a content image, a style reference image (such as an artwork by a famous painter), and the input image you want to style — and blend them together such that the input image is transformed to look like the content image, but “painted” in the style of the style image.

## NN Structure
 ![alt text](https://raw.githubusercontent.com/username/projectname/branch/path/to/img.png)

In the image-wise NN, convolutional layers and maxpooling are typically used. Usually, pre-trained networks with large datasets --such as VGG16 & VGG19-- are used. These networks are useful since they have been trained to extract features of the input images.

The first layers extract the most detailed features of the input image. On the other hand, the last layers contain main features (such as ears, mouth, etc.). The deepest the layer is chosen, the more the style will be used from that input image. Alternatively, if the chosen layer is extracting low-level features, the content will be more important

(REVIEW & REWRITE THIS LAST TEXT)
LOW LEVEL FEATURES / HIGH LEVEL FEATURES

## Gram Matrix
The Gram Matrix is used to compare both the style image and the output one.

The style representation of an image is described as the correlation of the different filter responses given by the Gram matrix.

Given the first layer of the trained network and a CxHxW vector space is obtained, where C is the number of filters, H is the height of the image and W the width. From these parameters, we compute the Gram Matrix. To obtain it, different rows are chosen and their inner product computed in order to see which neurons tend to be activated at the same time.


(FORMULA G^l_ij)
"where Gˡᵢⱼ is the inner product between the vectorized feature map i and j in layer l. We can see that Gˡᵢⱼ generated over the feature map for a given image represents the correlation between feature maps i and j."


## Loss functions
The principle of neural style transfer is to define two distance functions, one that describes how different the content of two images are, Lcontent, and one that describes the difference between the two images in terms of their style, Lstyle. Then, given three images, a desired style image, a desired content image, and the input image (initialized with the content image), we try to transform the input image to minimize the content distance with the content image and its style distance with the style image. (PREGUNTAR CRISITAN SI HAY 3 IMAGENES!)
In summary, we’ll take the base input image, a content image that we want to match, and the style image that we want to match. We’ll transform the base input image by minimizing the content and style distances (losses) with backpropagation, creating an image that matches the content of the content image and the style of the style image. (?)

In this case, the loss function will be formed by the content-image loss function --which represents how far is the generated image from the content one-- and the style-image loss function --which represents how well the style has been emulated--. (FORMULASSSS!).
