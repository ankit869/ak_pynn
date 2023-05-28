
# MultiLayer Neural Network (ak_pynn)
A simplistic and efficient pure-python neural network library, which can be used to build , visualize and deploy deep learning ANN models. It is optimized for best performance.

- Optimized for performance
- Better Visualization
- Cross platform



## Authors

- [@ankit_kohli](https://www.github.com/ankit869)


## License

[MIT](https://choosealicense.com/licenses/mit/)


## Support

For support, email contact.ankitkohli@gmail.com


## Features

- [x] Efficient implementations of activation functions and their gradients
    - [x]  Sigmoid
    - [x]  ReLU
    - [x]  Leaky ReLU
    - [x]  Softmax  
    - [x]  Softplus  
    - [x]  Tanh 
    - [x]  Elu  
    - [x]  Linear 
- [x] Efficient implementations of loss functions and their gradients
    - [x]  Mean squared error 
    - [x]  Mean absolute error
    - [x]  Binary cross entropy  
    - [x]  Categorical cross entropy  
- [x] Several methods for weights initialization
    - [x]  ```'random uniform'```, ```'random normal'```
    - [x]  ```'Glorot Uniform'```, ```'Glorot Normal'```
    - [x]  ```'He Uniform'```,```'He Normal'```

- [x] Neural network optimization using 
    - [x]  Gradient Descent (Batch/ SGD / Mini-Batch)
    - [x]  Momentum
    - [x]  Adagrad
    - [x]  RMSprop
    - [x]  Adam

- [x] Regularizations
    - [x]  L1 Norm
    - [x]  L2 Norm
    - [x]  L1_L2 Norm
    - [x]  Dropouts

- [x] Batch Normalization
- [x] Early Stopping
- [x] Validation Splits
- [x] Predict Scores
## Installation
Install the release (stable) version from PyPi
```
pip install ak-pynn
```

## Usage/Examples

Import
```python
from ak_pynn.mlp import MLP
```

Usage 
```python
model = MLP()
model.add_layer(4,input_layer=True)
model.add_layer(10,activation_function='relu',batch_norm=True)
model.add_layer(10,activation_function='relu',dropouts=True)
model.add_layer(10,activation_function='relu')
model.add_layer(3,activation_function='softmax',output_layer=True)
model.compile_model(optimizer='Adam',loss_function='mse',metrics=['mse','accuracy'])
```
Output
```

                                ( MODEL SUMMARY )                        
        
        ===================================================================
               Layer           Activation    Output Shape      Params    
        ===================================================================

               Input             linear       (None, 4)          0       
        -------------------------------------------------------------------

               Dense              relu        (None, 10)         50      
        -------------------------------------------------------------------

         BatchNormalization       None        (None, 10)         40      
        -------------------------------------------------------------------

               Dense              relu        (None, 10)        110      
        -------------------------------------------------------------------

              Dropout             None        (None, 10)         0       
        -------------------------------------------------------------------

               Dense              relu        (None, 10)        110      
        -------------------------------------------------------------------

               Output           softmax       (None, 3)          33      
        -------------------------------------------------------------------

        ===================================================================

        Total Params  - 343
        Trainable Params  - 323
        Non-Trainable Params  - 20
        ___________________________________________________________________
              
```
Visualizing model
```python
model.visualize()
```

![App Screenshot](https://drive.google.com/uc?id=1VHFYmo8ufV2_J0DuBvhipFNH1ezLQZIs)

Training the model
```python
model.fit(X_train, Y_train,epochs=200,batch_size=32,verbose=False,early_stopping=False,patience=3,validation_split=0.2)
model.predict_scores(X_test,Y_test,metrics=['accuracy','precision','macro_recall'])
plt.plot(model.history['Val_Losses'])
plt.plot(model.history['Losses'])

```
## TESTS


[@mnist_test](https://github.com/ankit869/ak_pynn/blob/main/mnist_test.ipynb)

[@iris_test](https://github.com/ankit869/ak_pynn/blob/main/iris_test.ipynb)

[@mlp_demo](https://github.com/ankit869/ak_pynn/blob/main/mlp_demo.ipynb)
## Citation
If you use this library and would like to cite it, you can use:
```
Ankit kohli, "ak-pynn: Neural Network libray", 2023. [Online]. Available: https://github.com/ankit869/ak-pynn. [Accessed: DD- Month- 20YY].
```
or:
```
@Misc{,
  author = {Ankit kohli},
  title  = {ak-pynn: Neural Network libray},
  month  = May,
  year   = {2023},
  note   = {Online; accessed <today>},
  url    = {https://github.com/ankit869/ak-pynn},
}
```