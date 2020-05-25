# ColorDeblurring
 CS 230 project involving the integration of deblurring and colorization CNNs
 
 
 Notes:
 The first implementation involves developing our own colorization CNN. The model was developed succesfully and different architecture sizes, number of iterations, iteration step size, optimizer, branch size, etc. were tested. However, due to the large quantity of images required to train the model, as well as the time it would take to train the CNN, we opted to use a pretrained colorization network, which can be seen in the second implementation.
 
 The second implementation involves merging a colorization CNN and a deblurring GAN. Both these networks required editing of the files in order to adjust the codes to suit our implementation. We tested first colorizing and then deblurring, and vice-versa. This was done in order to decide on the sequence of the Neural network implementation. Through our tests, we realized that deblurring followed by colorization is the better sequence of the two. Our current tasks are to implement an additional layer to the pre-trained deblurring network, (as well as possibly the colorization network) in order to better adjust to greyscaled deblurred images.
