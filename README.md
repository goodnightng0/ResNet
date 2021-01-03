# ResNet
Pytorch implementation of [“Identity Mappings in Deep Residual Networks”](https://arxiv.org/pdf/1603.05027.pdf)
</br>
Implement a classifier for the CIFAR-10 dataset with ResNet

#### Network Architecture
The Network consists of a 3\*3 conv, 4 stages(each having a set of blocks as shown in the following figures), average pooling and a fully connected layer.
<div>
  <img height="250" src="https://github.com/goodnightng0/ResNet/blob/main/architecture/stage1.PNG">
  <img height="250" src="https://github.com/goodnightng0/ResNet/blob/main/architecture/stage2.PNG">
  <img height="250" src="https://github.com/goodnightng0/ResNet/blob/main/architecture/stage34.PNG">
</div>

- With "skip connections", this architecture improves the ability to model identity function
- That is, with all convolutional weights set to 0, the block can refer to the model identity

Details for the rest of the architecture can be found in the [decription file](./6.pdf)

#### Parameter Testing
Through trial and error of the [parameters](./params), change and find the best **batch size** and **learning rate**

Best accuracy can be found when batch size is 8 and learning rate is 0.005 with an overall accuracy of 78.91%
<div>
  <img height="300" src="https://github.com/goodnightng0/ResNet/blob/main/params/b%3D8%20l%3D0.005.PNG">
  </div>
