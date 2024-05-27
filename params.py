class ModelHyperParameters:
    """
    A simple example class to demonstrate basic class structure in Python.
    """

    def __init__(self, L, layer_dims, learning_rate=0.1):
        """
        Initializes a Model's Hyper Parameters
        :param L: The number of Layers for this model, excluding the input layer X
        :param layer_dims: A list of L+1 elements where the ith element is the
                           dimension of the layer. The input layer is included.
        """
        # The number of layers must be greater than or equal to 1
        assert L >= 1
        assert len(layer_dims) - 1 == L

        # The number of hidden units in a layer must be greater than 0
        for element in layer_dims:
            assert element > 0

        self.L = L
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
