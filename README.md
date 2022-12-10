# Next-Frame Prediction
Next-frame prediction is a new, promising field of research in computer vision, predicting possible future images by presenting historical image information. It provides extensive application value in robot decision-making and autonomous driving. 

- **Model** <br/>
The Convolutional LSTM architectures bring together time series processing and computer vision by introducing a convolutional recurrent cell in a LSTM layer. Using the Convolutional LSTM model, we explored next-frame prediction - the model of predicting what the next video frame will look like from a series of previous frames.
  * 1.1.	**<em>Fast Fourier Convolution</em>** <br/> 
  Vanilla convolutions in modern deep networks are known to operate locally and at fixed scale (e.g., the widely-adopted 3 × 3 kernels in image-oriented tasks). This causes low efficiency in connecting two distant locations in the network. In this work, we propose a novel convolutional operator dubbed as Fast Fourier convolution (FFC), which has the main hallmarks of non-local receptive fields and cross-scale fusion within the convolutional unit. According to the spectral convolution theorem in Fourier theory, point-wise update in the spectral domain globally affects all input features involved in Fourier transform, which sheds light on neural architectural design with non-local receptive fields. Idea is to convert the spatial features to spectral features — apply some operations — convert back to spatial features. Operations in spectral-domain indicate the receptive field of convolution to the full resolution of the input feature map [(Chi et al., 2020)](http://www.cs.toronto.edu/~nitish/unsupervised_video/). <br/>
  In this project, in order to improve the convolution layers, we replaced them with fast fourier convolutions(FFC). Furthermore, each ffc layer is followed by a self-attention layer.
  
- **Dataset** <br/>
The [Moving MNIST](http://www.cs.toronto.edu/~nitish/unsupervised_video/) dataset contains 10,000 video sequences, each consisting of 20 frames. In each video sequence, two digits move independently around the frame, which has a spatial resolution of 64×64 pixels. The digits frequently intersect with each other and bounce off the edges of the frame.

<p align="center">
  <img src="https://i.imgur.com/UYMTsw7.gif" />
</p>

- **Results** <br/>
<p align="center">
  <img src="https://lh3.googleusercontent.com/eODrBrXkcfyaMh7bwZqsXTe2cWrEmvvYcFguCXXeGo8LVVB5CayaRPEndj7lW9zS9YU=w2400" />
</p>
As you can see, The best result was achieved when we used ConvLSTM with a  self-attention layer. The second-best result is the base model, which just uses ConvLSTM cells. We got the worst result when we used FFC instead of simple convolution layers. This is probably because of the complexity of the FfcLSTM model. As you can see in the chart above, the validation loss of this model is worse than the loss on the train set. It seems that this model has been overfitted on the train set.
