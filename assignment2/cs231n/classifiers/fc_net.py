from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        for l in range(1, self.num_layers):
            if l == 1:
                prev_dim = input_dim
            else:
                prev_dim = hidden_dims[l - 2]
            self.params['W' + str(l)]\
                = np.random.normal(0, weight_scale, (prev_dim, hidden_dims[l - 1]))
            self.params['b' + str(l)] = np.zeros((hidden_dims[l - 1], )) 

            if self.normalization is None:
                continue
            if l == self.num_layers - 1:
                break 
            self.params['gamma' + str(l)] = np.ones((hidden_dims[l - 1], )) 
            self.params['beta' + str(l)] = np.zeros((hidden_dims[l - 1], )) 
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        N = X.shape[0]
        X_2d = X.reshape((N, -1))
        aff_outs, norm_outs, relu_outs, drop_outs = [], [], [], []
        aff_caches, norm_caches, relu_caches, drop_caches = [], [], [], []

        out = X
        # not include the last hidden layer, L - 1 hidden layer in total
        for l in range(1, self.num_layers - 1):
            out, aff_cache = affine_forward(out, \
                self.params['W' + str(l)], self.params['b' + str(l)])
            aff_outs.append(out)
            aff_caches.append(aff_cache)

            if self.normalization == "batchnorm":
                out, norm_cache = batchnorm_forward(out, \
                    self.params['gamma' + str(l)], 
                    self.params['beta' + str(l)], 
                    self.bn_params[l - 1])
                norm_outs.append(out)
                norm_caches.append(norm_cache)
            elif self.normalization == "layernorm":
                out, norm_cache = layernorm_forward(out, \
                    self.params['gamma' + str(l)], 
                    self.params['beta' + str(l)], 
                    self.bn_params[l - 1])
                norm_outs.append(out)
                norm_caches.append(norm_cache)

            out, relu_cache = relu_forward(out)
            relu_outs.append(out)
            relu_caches.append(relu_cache)

            if self.use_dropout:
                out, drop_cache = dropout_forward(out, self.dropout_param)
                drop_outs.append(out)
                drop_caches.append(drop_cache)

        
        # last hidden layer
        scores, cache = affine_forward(out, \
            self.params['W' + str(self.num_layers - 1)], self.params['b' + str(self.num_layers - 1)])
        aff_caches.append(cache) 
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)
        dX, dW, db = affine_backward(dscores, aff_caches[-1])   
        grads['W' + str(self.num_layers - 1)] = \
            dW + self.reg * self.params['W' + str(self.num_layers - 1)]
        grads['b' + str(self.num_layers - 1)] = db       
        
        # for all the first L - 1 hidden layers
        for l in range(self.num_layers - 2, 0, -1):
            if self.use_dropout:
                dX = dropout_backward(dX, drop_caches[l - 1])

            dX = relu_backward(dX, relu_caches[l - 1])

            if self.normalization == 'batchnorm':
                dX, dgamma, dbeta = batchnorm_backward(dX, norm_caches[l - 1])
                grads['gamma' + str(l)] = dgamma
                grads['beta' + str(l)] = dbeta
            elif self.normalization == 'layernorm':
                dX, dgamma, dbeta = layernorm_backward(dX, norm_caches[l - 1])
                grads['gamma' + str(l)] = dgamma
                grads['beta' + str(l)] = dbeta
            
            dX, dW, db = affine_backward(dX, aff_caches[l - 1])
            grads['W' + str(l)] = \
                dW + self.reg * self.params['W' + str(l)]
            grads['b' + str(l)] = db 

        reg_term = 0
        for l in range(1, self.num_layers):
            reg_term += np.sum(self.params['W' + str(l)] ** 2)
        reg_term *= 0.5 * self.reg
        loss += reg_term 
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
    
    def save(self, fname):
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
      np.save(fpath, params)
      print(fname, "saved.")
    
    def load(self, fname):
      """Load model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "loaded.")
        return True
