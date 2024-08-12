<h3 align="left">
My first Neural Network built from scratch using only numpy.
</h3>

## Motivation

In 2020-2021 when I was a DDoS researcher at IFC I evaluated different machine learning algorithms for network traffic classification and DoS + DDoS attack detection. I used libraries like scikit-learn and tensorflow to build and train models, and was able to achieve good results with the simple models based on logistic regressions, random forests and support vector machines. This models were more interpretable and easier to understand, so I had the feeling that I knew what I was doing. When I tried to use neural networks it also worked, but I had no idea how and why it was working, I was just using scikit-learn and tensorflow as black boxes.

After the transformers and GPT boom I finally decided to dedicate some time to understand how neural networks work. I read some articles and watched some really good youtube videos but in the end I had to get my hands dirty to really understand it. I tried going straight to the code, but I was lost - I had no fundamentals, didn't even remember matrix multiplication. So I sharpened the pencil with some linear algebra and calculus and then finally was able to build something more concrete.

## About the project

This project is a hello world for neural networks. It is a simple implementation with 3 layers (input, hidden and output) and a sigmoid activation function. This neural network is trained to identify handwritten digits from 0 to 9 displayed in a 28x28 pixel image (784 input nodes, each representing a pixel value and 10 output nodes representing the numbers 0 to 9). The weights are initialized randomly from a normal distribution centered in 0 and the training is done using the backpropagation algorithm. The optimization is done using the gradient descent algorithm with a fixed learning rate.

The dataset used for training is the MNIST dataset, which contains 60,000 training images and 10,000 test images. The model is trained for 10 epochs and for each image we rotate the image 10 degrees clockwise and 10 degrees counterclockwise to improve the dataset, so the model trains 3 times for each record in every epoch. The model is evaluated using the test dataset and the accuracy is calculated.

The scores varies because of the random weights, but the model is able to consistently achieve a score of 95% to 97% accuracy, which is not bad for a very simple neural network.

This project fulfilled its purpose of helping me understand how computers can emulate our neurons and how we can build neural networks. It was a very rewarding project and I am very happy with the results. Now that I have a very good understanding of the fundamentals, I can move on to more complex models and architectures - the next steps are convolutional neural networks and transformers.

## How to run

After cloning this repository, download the MNIST datasets:

- Train dataset: http://www.pjreddie.com/media/files/mnist_train.csv
- Test dataset: http://www.pjreddie.com/media/files/mnist_test.csv

Install the dependencies in your environment:

```bash
pip install -r requirements.txt
```

or

```bash
conda install --file requirements.txt
```

This assumes that you have jupyter notebook installed and running.

Then open the jupyter notebook and run the cells. The main model is in the `neural_network.ipynb` file. The `backtrack.ipynb` file is a playground for receiving a number from 0 to 9 as input and the trained neural network will try to build the 28x28 image of that number.

## References

- [3Blue1Brown Neural Network Playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [Make Your Own Neural Network, by Tariq Rashid](https://www.amazon.com/Make-Your-Own-Neural-Network-ebook/dp/B01EER4Z4G)
- [Yann LeCun's MNIST page](http://yann.lecun.com/exdb/mnist/)

###### Vitor Matheus Valandro da Rosa. July 2024.
