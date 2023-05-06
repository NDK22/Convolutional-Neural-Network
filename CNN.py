import tensorflow as tf
import numpy as np
import keras
#import tensorflow.keras as keras

#Nelson can you read?

class CNN(object):
    def __init__(self):
        self.model= keras.models.Sequential()
        self.metric = []
        self.layer_map={}
        self.loss = None
        self.optimizer = None
        """
        Initialize multi-layer neural network

        """


    def add_input_layer(self, shape=(2,),name="" ):
        if len(self.model.layers) > 0 and isinstance(self.model.layers[0],tf.keras.layers.InputLayer):
            self.model.input_shape=shape
            self.model.name = name
        self.model.add(tf.keras.layers.InputLayer(input_shape=shape,name=name))
        return None
        """
        This method adds an input layer to the neural network. If an input layer exist, then this method
         should replace it with the new input layer.
         Input layer is considered layer number 0, and it does not have any weights. Its purpose is to determine
         the shape of the input tensor and distribute it to the next layer.
         :param shape: input shape (tuple)
         :param name: Layer name (string)
         :return: None
         """



    def append_dense_layer(self, num_nodes,activation="relu",name="",trainable=True):
        activation = activation.lower()
        if activation not in ["linear", "relu", "sigmoid","softmax"]:
            raise ValueError("Invalid Activation Function")
        dense_layer = tf.keras.layers.Dense(units=num_nodes,activation=activation,name=name,trainable=trainable)
        self.model.add(dense_layer)
        return None
        """
         This method adds a dense layer to the neural network
         :param num_nodes: Number of nodes
         :param activation: Activation function for the layer. Possible values are "Linear", "Relu", "Sigmoid",
         "Softmax"
         :param name: Layer name (string)
         :param trainable: Boolean
         :return: None
         """
    def append_conv2d_layer(self, num_of_filters, kernel_size=3, padding='same', strides=1,
                         activation="Relu",name="",trainable=True):
        activation = activation.lower()
        if activation not in ["linear", "relu", "sigmoid","softmax"]:
            raise ValueError("Invalid Activation Function")
        layer_defined = tf.keras.layers.Conv2D(filters=num_of_filters,kernel_size=kernel_size,padding=padding,strides=strides,activation=activation,name=name,trainable=trainable)
        self.model.add(layer_defined)
        return layer_defined
        """
         This method adds a conv2d layer to the neural network
         :param num_of_filters: Number of nodes
         :param num_nodes: Number of nodes
         :param kernel_size: Kernel size (assume that the kernel has the same horizontal and vertical size)
         :param padding: "same", "Valid"
         :param strides: strides
         :param activation: Activation function for the layer. Possible values are "Linear", "Relu", "Sigmoid"
         :param name: Layer name (string)
         :param trainable: Boolean
         :return: Layer object
         """
    def append_maxpooling2d_layer(self, pool_size=2, padding="same", strides=2,name=""):
        layer_defined = tf.keras.layers.MaxPooling2D(pool_size=pool_size,padding=padding,strides=strides,name=name)
        self.model.add(layer_defined)
        return layer_defined
    
        """
         This method adds a maxpool2d layer to the neural network
         :param pool_size: Pool size (assume that the pool has the same horizontal and vertical size)
         :param padding: "same", "valid"
         :param strides: strides
         :param name: Layer name (string)
         :return: Layer object
         """
    def append_flatten_layer(self,name=""):
        layer_defined = tf.keras.layers.Flatten(name=name)
        self.model.add(layer_defined)
        return layer_defined
        """
         This method adds a flattening layer to the neural network
         :param name: Layer name (string)
         :return: Layer object
         """
    def set_training_flag(self,layer_numbers=[],layer_names="",trainable_flag=True):
        if not layer_numbers and not layer_names:
            raise ValueError("At least one of the layer_numbers or layer_names should be provided")
        
        if isinstance(layer_names,str):
            layer_names = [layer_names]
        
        if layer_numbers:
            for layer_number in layer_numbers:
                if layer_number < len(self.model.layers):
                    self.model.get_layer(index=layer_number).trainable = trainable_flag
                else:
                    raise ValueError(f"Layer with number: 'layer_number' not found")
        
        if layer_names:
            for layer_name in layer_names:
                if layer_name in self.layer_map:
                        self.model.get_layer(name=layer_name).trainable = trainable_flag
                else:
                    raise ValueError(f"Layer with name: 'layer_name' not found") 
                self.layer_map[layer_name]=trainable_flag              
        
        return None
        """
        This method sets the trainable flag for a given layer
        :param layer_number: an integer or a list of numbers.Layer numbers start from layer 0.
        :param layer_names: a string or a list of strings (if both layer_number and layer_name are specified, layer number takes precedence).
        :param trainable_flag: Set trainable flag
        :return: None
        """

    def get_weights_without_biases(self,layer_number=None,layer_name=""):
        if layer_number is not None and layer_name != "":
            if layer_number >= 0:
                layers_weights= self.model.get_layer(index=layer_number-1)
            else:
                layers_weights= self.model.get_layer(index=layer_number)
        elif layer_number is not None:
            if layer_number >= 0:
                layers_weights= self.model.get_layer(index=layer_number-1)
            else:
                layers_weights= self.model.get_layer(index=layer_number)
        elif layer_name != "":
            layers_weights= self.model.get_layer(name=layer_name)
        else:
            raise ValueError("Provide layer number or layer name")
        
        if len(layers_weights.get_weights()) > 0 and layer_number != 0:
            weight_matrix = layers_weights.get_weights()[0]
            return weight_matrix
        else:
            return None

        """
        This method should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero. Note that layer 0 is the input layer.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0. Note that layer 0 is the input layer
         and it does not have any weights or biases.
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: Weight matrix for the given layer (not including the biases). If the given layer does not have
          weights then None should be returned.
         """


    def get_biases(self,layer_number=None,layer_name=""):
        if layer_number is not None and layer_name != "":
            if layer_number >= 0:
                layers_biases= self.model.get_layer(index=layer_number-1)
            else:
                layers_biases= self.model.get_layer(index=layer_number)
        elif layer_number is not None:
            if layer_number >= 0:
                layers_biases= self.model.get_layer(index=layer_number-1)
            else:
                layers_biases= self.model.get_layer(index=layer_number)
        elif layer_name != "":
            layers_biases= self.model.get_layer(name=layer_name)
        else:
            raise ValueError("Provide layer number or layer name")
        bias_matrix = layers_biases.get_weights()[1] if len(layers_biases.get_weights()) > 0 and layer_number != 0 else None
        return bias_matrix
        
        """
        This method should return the biases for layer layer_number.
        layer numbers start from zero. Note that layer 0 is the input layer.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0.Note that layer 0 is the input layer
         and it does not have any weights or biases.
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: biases for the given layer (If the given layer does not have bias then None should be returned)
         """

    def set_weights_without_biases(self,weights,layer_number=None,layer_name=""):
        if layer_number is not None and layer_name != "":
            if layer_number >= 0:
                layers_weights= self.model.get_layer(index=layer_number-1)
            else:
                layers_weights= self.model.get_layer(index=layer_number)
        elif layer_number is not None:
            if layer_number >= 0:
                layers_weights= self.model.get_layer(index=layer_number-1)
            else:
                layers_weights= self.model.get_layer(index=layer_number)
        elif layer_name != "":
            layers_weights= self.model.get_layer(name=layer_name)
        else:
            raise ValueError("Provide layer number or layer name")
        layers_weights.set_weights([weights,layers_weights.get_weights()[1]])
        return None

        """
        This method sets the weight matrix for layer layer_number.
        layer numbers start from zero. Note that layer 0 is the input layer.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0. Note that layer 0 is the input layer
         and it does not have any weights or biases.
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: None
         """
    def set_biases(self,biases,layer_number=None,layer_name=""):
        if layer_number is not None and layer_name != "":
            if layer_number >= 0:
                layers_biases= self.model.get_layer(index=layer_number-1)
            else:
                layers_biases= self.model.get_layer(index=layer_number)
        elif layer_number is not None:
            if layer_number >= 0:
                layers_biases= self.model.get_layer(index=layer_number-1)
            else:
                layers_biases= self.model.get_layer(index=layer_number)
        elif layer_name != "":
            layers_biases= self.model.get_layer(name=layer_name)
        else:
            raise ValueError("Provide layer number or layer name")
        layers_biases.set_weights([layers_biases.get_weights()[0],biases])
        return None
        """
        This method sets the biases for layer layer_number.
        layer numbers start from zero. Note that layer 0 is the input layer.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0. Note that layer 0 is the input layer
         and it does not have any weights or biases.
        :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
        :return: none
        """
    def remove_last_layer(self):
        remove_end_layer = self.model.pop()
        self.model = keras.models.Sequential(self.model.layers)
        return remove_end_layer
        """
        This method removes a layer from the model.
        :return: removed layer
        """

    def load_a_model(self,model_name="",model_file_name=""):
        self.model = keras.models.Sequential()
        model_name = model_name.lower()
        if model_name:
            if model_name == 'vgg16':
                model = tf.keras.applications.VGG16()
            elif model_name == 'vgg19':
                model = tf.keras.applications.VGG19()
            else:
                raise ValueError("Invalid Model Name")
        elif model_file_name:
            model=tf.keras.models.load_model(model_file_name)
        else:
            raise ValueError("Provide with model name or model file name")
        for layer in model.layers:
          self.model.add(layer)
        return self.model
        """
        This method loads a model architecture and weights.
        :param model_name: Name of the model to load. model_name should be one of the following:
        "VGG16", "VGG19"
        :param model_file_name: Name of the file to load the model (if both madel_name and
         model_file_name are specified, model_name takes precedence).
        :return: model
        """
    def save_model(self,model_file_name=""):
        if not model_file_name:
            raise ValueError("Please provide with the model file name")
        self.model.save(model_file_name)
        return self.model
        """
        This method saves the current model architecture and weights together in a HDF5 file.
        :param file_name: Name of file to save the model.
        :return: model
        """


    def set_loss_function(self, loss="SparseCategoricalCrossentropy"):
        if loss not in ["SparseCategoricalCrossentropy","MeanSquaredError","hinge"]:
            raise ValueError("Invalid Loss type. Please choose from 'SparseCategoricalCrossentropy', 'MeanSquaredError', or 'hinge'")
        self.loss= loss
        return None
        """
        This method sets the loss function.
        :param loss: loss is a string with the following choices:
        "SparseCategoricalCrossentropy",  "MeanSquaredError", "hinge".
        :return: none
        """

    def set_metric(self,metric):
        if metric not in ["accuracy", "mse"]:
            raise ValueError("Invalid Metric. Please choose from 'accuracy', or 'mse'")
        self.metric=metric
        return None
        """
        This method sets the metric.
        :param metric: metric should be one of the following strings:
        "accuracy", "mse".
        :return: none
        """

    def set_optimizer(self,optimizer="SGD",learning_rate=0.01,momentum=0.0):
        optimizer = optimizer.lower()
        if optimizer == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,momentum=momentum)
        elif optimizer == "rmsprop":
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate,momentum=momentum)
        elif optimizer == "adagrad":
            self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        else:
            raise ValueError("Invalid Optimizer. Please choose from 'SGD', 'RMSprop' or 'Adagrad'")
        return None
        """
        This method sets the optimizer.
        :param optimizer: Should be one of the following:
        "SGD" , "RMSprop" , "Adagrad" ,
        :param learning_rate: Learning rate
        :param momentum: Momentum
        :return: none
        """

    def predict(self, X):
        input = tf.convert_to_tensor(X,dtype=tf.float32)
        output = self.model.predict(x=input)
        return output
        """
        Given array of inputs, this method calculates the output of the multi-layer network.
        :param X: Input tensor.
        :return: Output tensor.
        """

    def evaluate(self,X,y):
        input = tf.convert_to_tensor(X,dtype=tf.float32)
        output = tf.convert_to_tensor(y,dtype=tf.float32)
        loss, metric = self.model.evaluate(x=input,y=output)
        return loss, metric

        """
         Given array of inputs and desired ouputs, this method returns the loss value and metrics of the model.
         :param X: Array of input
         :param y: Array of desired (target) outputs
         :return: loss value and metric value
         """
    def train(self, X_train, y_train, batch_size, num_epochs):
        X_train = tf.convert_to_tensor(X_train,dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train,dtype=tf.float32)
        self.model.compile(optimizer=self.optimizer,loss=self.loss,metrics=self.metric)
        history = self.model.fit(x=X_train,y=y_train,batch_size=batch_size,epochs=num_epochs)
        loss_list = history.history['loss']
        return loss_list
        """
         Given a batch of data, and the necessary hyperparameters,
         this method trains the neural network by adjusting the weights and biases of all the layers.
         :param X_train: Array of input
         :param y_train: Array of desired (target) outputs
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :return: list of loss values. Each element of the list should be the value of loss after each epoch.
         """