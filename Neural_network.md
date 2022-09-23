## Neural network

### Concept

### Applications

Object identification

### Components

Neural networks consists of four key components
- Network: Calculate parameters(weight,bias) using layers and nodes
- Activation: Convert result of neural network to final output
- Loss calculation: Calculate the difference between true value and predicted value.
- Optimization: Update parameters in the network to minimize loss

| Purpose | PyTorch | Keras |
|---|---|---|
|**Loss function**|||
| Binary classification | `nn.BELoss` | `losses.binary_crossentropy` |
| Multiclass classification | `nn.CrossEntroypLoss` | `losses.categorical_crossentropy` |
| Regression | `nn.MSELoss` | `losses.mean_squared_error` |
|**Activation function**|
| Binary classification | `nn.Sigmoid` | `activations.sigmoid` |
| Multiclass classification | `nn.Softmax` | `activations.softmax` |
|**Optimization function**| | |
| Adam | `optim.Adam` | `optimizers.Adam` |
| Stochastic Gradient Descent | `optim.SGD` | `optimizers.SGD` |

### Tools
    - [PyTorch](https://pytorch.org/) backed by Meta
    - [Keras/Tensorflow](https://www.tensorflow.org/) backed by Google

### Architecture
    - CNN
    - RNN
    - ...

### Popular models
    - YoLo
    
### Example code



