# This project show cases the use of GANs

In this project we've tried building GAN composed of a discriminator, which is a CNN model <br />
which takes an image as an input and tells how close it is to being a photograph as a floating point score. <br />
Then we have an inverse CNN (or upsampling) model which takes a floating point score and tries to generate <br />
a photograph based on that score. <br />
<br />

Now these two models are coupled together in an adverserial setting to achieve a model that could generate <br />
a photograph out of human sketch faces.
