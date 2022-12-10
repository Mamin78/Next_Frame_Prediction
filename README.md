# Next-Frame Prediction
Next-frame prediction is a new, promising field of research in computer vision, predicting possible future images by presenting historical image information. It provides extensive application value in robot decision-making and autonomous driving. 

- **Model** <br/>
The Convolutional LSTM architectures bring together time series processing and computer vision by introducing a convolutional recurrent cell in a LSTM layer. Using the Convolutional LSTM model, we explored next-frame prediction - the model of predicting what the next video frame will look like from a series of previous frames.

- **Dataset** <br/>
The [Moving MNIST](http://www.cs.toronto.edu/~nitish/unsupervised_video/) dataset contains 10,000 video sequences, each consisting of 20 frames. In each video sequence, two digits move independently around the frame, which has a spatial resolution of 64×64 pixels. The digits frequently intersect with each other and bounce off the edges of the frame.

<p align="center">
  <img src="https://i.imgur.com/UYMTsw7.gif" />
</p>
