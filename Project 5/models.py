import nn
import numpy as np

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** CS5368 YOUR CODE HERE ***"
        weights = self.get_weights()
        
        dotProduct = nn.DotProduct(weights, x)
        return dotProduct

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** CS5368 YOUR CODE HERE ***"
        # Calculates score for input value with assigned weight value using DotProduct method
        run_val = self.run(x)
        
        # As per instruction, i used scalar method to convert node value to numeric value
        scalar_value = nn.as_scalar(run_val)
        
        # If our calculated value is greater than 0, it means we have predicted class value for particular input and return 1, else return -1
        if scalar_value >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** CS5368 YOUR CODE HERE ***"
        
        # To classifies the input data, i have executed loop for infinie time
        while True:
            flag = 1
            
            # Get data value using iterate_once method from Dataset class which includes Input(x) value and Expected Output class(y) value
            dataset_iteration = dataset.iterate_once(1)
            
            # Iterate through database for each input and output iteration
            for x, y in dataset_iteration:
                
                # Predict input data using above get_prediction method which return predicted class for input
                prediction = self.get_prediction(x)
                
                # If Predicted value and expected output class value is not same, then we have to recalculate value again
                if prediction != nn.as_scalar(y):
                    # To recalculate weight, as per given instruction, i used update method to calculate weight (weights←weights+direction⋅multiplier)
                    self.w.update(x,nn.as_scalar(y))
                    flag = 0
            
            # If flag = 1, it means our all data values get classified properly, so we exit the loop
            if flag == 1:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** CS5368 YOUR CODE HERE ***"
        
        self.batch_size = 1
        
        # I used 2 feature and bias value for RegressionModel with dimeansion of 50
        self.feature0 = nn.Parameter(1,50)
        self.bias0 = nn.Parameter(1,50)
        
        self.feature1 = nn.Parameter(50,1)
        self.bias1 = nn.Parameter(1,1)

        self.learning_rate = 0.005

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** CS5368 YOUR CODE HERE ***"
        
        # Before adding bias to each vector, we need to transform each input to give value in Batch Size X Output Feature
        calculateLinear1 = nn.Linear(x, self.feature0)
        # Add Linear Transformed Input to Bias Vector
        addBias1 = nn.AddBias(calculateLinear1, self.bias0)
        # Relu calculates node with same shape as input using bias value
        relu1 = nn.ReLU(addBias1)
        
        # We have 2 feature input, so we are calculating ReLU unit for 2 times
        # Before adding bias to each vector, we need to transform each input to give value in Batch Size X Output Feature
        calculateLinear2 = nn.Linear(relu1, self.feature1)
        # Add Linear Transformed Input to Bias Vector
        addBias2 = nn.AddBias(calculateLinear2, self.bias1)
        
        return addBias2

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** CS5368 YOUR CODE HERE ***"
        
        # To calculate loss, we have given SquareLoss metod in comment, so i used SquareLoss method to calculate loss by running model
        runModel = self.run(x)
        # Calculate Loss for Executed Model
        squareLoss = nn.SquareLoss(runModel, y)
        
        return squareLoss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** CS5368 YOUR CODE HERE ***"
        
        # To train model, i used same technique which, i used in PerceptronModel. I am executing model for infinite time, till i don't get loss less than 0.02
        
        while True:
            dataset_iteration = dataset.iterate_once(1)
            
            # Iterate through database for each input and output iteration
            for x, y in dataset_iteration:
                # As per given steps in comments, I have calculated Loss Value
                loss = self.get_loss(x, y)
                
                # Then, We have to calculate Gradient value - nn.gradients
                parameters = [self.feature0, self.feature1, self.bias0,self.bias1]
                gradient = nn.gradients(loss, parameters)
                
                # To calculate weight, I used update method for each feature and bias used for model, but here, instead of passing scalar quantity of output, i passlearning rate
                self.feature0.update(gradient[0], -self.learning_rate)
                self.feature1.update(gradient[1], -self.learning_rate)
                self.bias0.update(gradient[2], -self.learning_rate)
                self.bias1.update(gradient[3], -self.learning_rate)
                
            # As per given suggestion in comments, i used nn.as_scalar method to stop training
            
            # Fetch Constant values for Input and Output values
            xConstant = nn.Constant(dataset.x)
            yConstant = nn.Constant(dataset.y)
            
            # calculate loss values for constant input and output
            lossValue = self.get_loss(xConstant, yConstant)
            
            if nn.as_scalar(lossValue) < 0.02:
                return

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** CS5368 YOUR CODE HERE ***"
        
        # I refer same initialization as mentioned in RegressionModel, just updated their dimension to 784
        self.batch_size = 1
        
        # I used 2 feature and bias value for RegressionModel with dimeansion of 50
        self.feature0 = nn.Parameter(784,150)
        self.bias0 = nn.Parameter(1,150)
        
        self.feature1 = nn.Parameter(150,10)
        self.bias1 = nn.Parameter(1,10)

        self.learning_rate = 0.005

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** CS5368 YOUR CODE HERE ***"
        
        # Used same logic as RegressionModel
        
        # Before adding bias to each vector, we need to transform each input to give value in Batch Size X Output Feature
        calculateLinear1 = nn.Linear(x, self.feature0)
        # Add Linear Transformed Input to Bias Vector
        addBias1 = nn.AddBias(calculateLinear1, self.bias0)
        # Relu calculates node with same shape as input using bias value
        relu1 = nn.ReLU(addBias1)
        
        # We have 2 feature input, so we are calculating ReLU unit for 2 times
        # Before adding bias to each vector, we need to transform each input to give value in Batch Size X Output Feature
        calculateLinear2 = nn.Linear(relu1, self.feature1)
        # Add Linear Transformed Input to Bias Vector
        addBias2 = nn.AddBias(calculateLinear2, self.bias1)
        
        return addBias2

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** CS5368 YOUR CODE HERE ***"

        # To calculate loss, we have given SoftmaxLoss metod in comment, so i used SoftmaxLoss method to calculate loss by running model
        runModel = self.run(x)
        # Calculate Loss for Executed Model
        softMaxLoss = nn.SoftmaxLoss(runModel, y)
        
        return softMaxLoss


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** CS5368 YOUR CODE HERE ***"
        
        # To train model, i used same technique which, i used in PerceptronModel. I am executing model for infinite time, till i don't get loss less than 0.02
        
        while True:
            dataset_iteration = dataset.iterate_once(1)
            
            # Iterate through database for each input and output iteration
            for x, y in dataset_iteration:
                # As per given steps in comments, I have calculated Loss Value
                loss = self.get_loss(x, y)
                
                # Then, We have to calculate Gradient value - nn.gradients
                parameters = [self.feature0, self.feature1, self.bias0,self.bias1]
                gradient = nn.gradients(loss, parameters)
                
                # To calculate weight, I used update method for each feature and bias used for model, but here, instead of passing scalar quantity of output, i passlearning rate
                self.feature0.update(gradient[0], -self.learning_rate)
                self.feature1.update(gradient[1], -self.learning_rate)
                self.bias0.update(gradient[2], -self.learning_rate)
                self.bias1.update(gradient[3], -self.learning_rate)
                
            # As per given suggestion in comments, i used dataset.get_validation_accuracy() method to calculate accuracy and stop training
            if dataset.get_validation_accuracy() >= 0.97:
                return


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** CS5368 YOUR CODE HERE ***"
        
        # I refer same initialization as mentioned in RegressionModel, just updated their dimension to 784
        self.batch_size = 1
        
        # I used 2 feature and bias value for RegressionModel with dimeansion of 50
        self.feature0 = nn.Parameter(47,150)
        self.bias0 = nn.Parameter(1,150)
        
        self.feature1 = nn.Parameter(150,150)
        self.bias1 = nn.Parameter(1,150)

        self.hidden_weight = nn.Parameter(150,150)
        self.weight = nn.Parameter(150, 5)

        self.learning_rate = 0.075

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** CS5368 YOUR CODE HERE ***"
        
        # Used same logic as RegressionModel
        
        # Before adding bias to each vector, we need to transform each input to give value in Batch Size X Output Feature
        calculateLinear1 = nn.Linear(xs[0], self.feature0)
        # Add Linear Transformed Input to Bias Vector
        addBias1 = nn.AddBias(calculateLinear1, self.bias0)
        # Relu calculates node with same shape as input using bias value
        relu1 = nn.ReLU(addBias1)
        
        # We have 2 feature input, so we are calculating ReLU unit for 2 times
        # Before adding bias to each vector, we need to transform each input to give value in Batch Size X Output Feature
        calculateLinear2 = nn.Linear(relu1, self.feature1)
        # Add Linear Transformed Input to Bias Vector
        addBias2 = nn.AddBias(calculateLinear2, self.bias1)
        
        for i in xs:
            # Calculate weight for feature0
            weight = nn.Linear(i, self.feature0)
            hidden_weight = nn.Linear(addBias2, self.hidden_weight)
            
            add = nn.Add(weight, hidden_weight)
            
            relu = nn.ReLU(add)
            
            addBias2 = nn.AddBias(relu, self.bias0)
            
        return nn.Linear(addBias2, self.weight)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** CS5368 YOUR CODE HERE ***"
        
        # To calculate loss, we have given SoftmaxLoss metod in comment, so i used SoftmaxLoss method to calculate loss by running model
        runModel = self.run(xs)
        # Calculate Loss for Executed Model
        softMaxLoss = nn.SoftmaxLoss(runModel, y)
        
        return softMaxLoss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** CS5368 YOUR CODE HERE ***"
        # To train model, i used same technique which, i used in PerceptronModel. I am executing model for infinite time, till i don't get loss less than 0.02
        
        while True:
            dataset_iteration = dataset.iterate_once(1)
            
            # Iterate through database for each input and output iteration
            for x, y in dataset_iteration:
                # As per given steps in comments, I have calculated Loss Value
                loss = self.get_loss(x, y)
                
                # Then, We have to calculate Gradient value - nn.gradients
                parameters = [self.feature0, self.feature1, self.bias0,self.bias1]
                gradient = nn.gradients(loss, parameters)
                
                # To calculate weight, I used update method for each feature and bias used for model, but here, instead of passing scalar quantity of output, i passlearning rate
                self.feature0.update(gradient[0], -self.learning_rate)
                self.feature1.update(gradient[1], -self.learning_rate)
                self.bias0.update(gradient[2], -self.learning_rate)
                self.bias1.update(gradient[3], -self.learning_rate)
                
            # As per given suggestion in comments, i used dataset.get_validation_accuracy() method to calculate accuracy and stop training
            if dataset.get_validation_accuracy() >= 0.80:
                return
