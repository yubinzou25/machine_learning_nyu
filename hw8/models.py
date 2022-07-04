import nn

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
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x)) < 0 :
            return -1
        else:
            return 1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        flag = True
        while flag:
            flag = False
            for x, y in dataset.iterate_once(1):
                if nn.as_scalar(y) != self.get_prediction(x):
                    self.w.update(x, nn.as_scalar(y))
                    flag = True




class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.param_list = []
        self.m1 = nn.Parameter(1, 10)
        self.param_list.append(self.m1)
        self.b1 = nn.Parameter(1, 10)
        self.param_list.append(self.b1)
        self.m2 = nn.Parameter(10, 10)
        self.param_list.append(self.m2)
        self.b2 = nn.Parameter(1, 10)
        self.param_list.append(self.b2)
        self.m3 = nn.Parameter(10, 1)
        self.param_list.append(self.m3)
        self.b3 = nn.Parameter(1, 1)
        self.param_list.append(self.b3)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        f1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.m1), self.b1))
        f2 = nn.ReLU(nn.AddBias(nn.Linear(f1, self.m2), self.b2))
        f3 = nn.AddBias(nn.Linear(f2, self.m3), self.b3)
        return f3
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_pred = self.run(x)

        return nn.SquareLoss(y_pred, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learning_rate = -0.01
        while True:
            total_loss = 0
            count = 0
            for x, y in dataset.iterate_once(5):
                loss = self.get_loss(x, y)
                total_loss += nn.as_scalar(loss)
                count += 1
                grad_wrt = nn.gradients(loss, self.param_list)
                for grad, param in zip(grad_wrt, self.param_list):
                    param.update(grad, learning_rate)
            print(total_loss/count)
            if total_loss/count < 0.02:
                break

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
        "*** YOUR CODE HERE ***"
        self.param_list = []
        self.m1 = nn.Parameter(784, 100)
        self.param_list.append(self.m1)
        self.b1 = nn.Parameter(1, 100)
        self.param_list.append(self.b1)
        self.m2 = nn.Parameter(100, 100)
        self.param_list.append(self.m2)
        self.b2 = nn.Parameter(1, 100)
        self.param_list.append(self.b2)
        self.m3 = nn.Parameter(100, 10)
        self.param_list.append(self.m3)
        self.b3 = nn.Parameter(1, 10)
        self.param_list.append(self.b3)

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
        "*** YOUR CODE HERE ***"
        f1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.m1), self.b1))
        f2 = nn.ReLU(nn.AddBias(nn.Linear(f1, self.m2), self.b2))
        f3 = nn.AddBias(nn.Linear(f2, self.m3), self.b3)
        return f3

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
        "*** YOUR CODE HERE ***"
        y_pred = self.run(x)

        return nn.SoftmaxLoss(y_pred, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learning_rate = -0.01
        while True:
            total_loss = 0
            count = 0
            for x, y in dataset.iterate_once(10):
                loss = self.get_loss(x, y)
                total_loss += nn.as_scalar(loss)
                count += 1
                grad_wrt = nn.gradients(loss, self.param_list)
                for grad, param in zip(grad_wrt, self.param_list):
                    param.update(grad, learning_rate)
            val_acc = dataset.get_validation_accuracy()
            print(val_acc)
            if val_acc > 0.975:
                break

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
        self.param_list = []
        self.w = nn.Parameter(self.num_chars, 64)
        self.param_list.append(self.w)
        self.w_hidden1 = nn.Parameter(64, 64)
        self.param_list.append(self.w_hidden1)
        self.w_hidden2 = nn.Parameter(64, 64)
        self.param_list.append(self.w_hidden2)
        self.w_final = nn.Parameter(64, 5)
        self.param_list.append(self.w_final)

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

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
        "*** YOUR CODE HERE ***"
        f_init = nn.ReLU(nn.Linear(xs[0], self.w))
        h = f_init
        for x in xs[1:]:
            h = nn.ReLU(nn.Linear(h, self.w_hidden1))
            z = nn.Add(nn.Linear(x, self.w), nn.Linear(h, self.w_hidden2))
            h = z
        return nn.Linear(h, self.w_final)

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
        "*** YOUR CODE HERE ***"
        y_pred = self.run(xs)
        return nn.SoftmaxLoss(y_pred, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learning_rate = -0.01
        while True:
            total_loss = 0
            count = 0
            for x, y in dataset.iterate_once(10):
                loss = self.get_loss(x, y)
                total_loss += nn.as_scalar(loss)
                count += 1
                grad_wrt = nn.gradients(loss, self.param_list)
                for grad, param in zip(grad_wrt, self.param_list):
                    param.update(grad, learning_rate)
            val_acc = dataset.get_validation_accuracy()
            if val_acc > 0.85:
                break
