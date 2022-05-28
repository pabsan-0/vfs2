
def onehot_enable(method):
    """ Decorator for the __call__ method inside mi_function to allow one-hot
    encoded features. Chooses the onehot encoded children features from a list
    with the names of their parents, relying on the separator character '#'.
        > Expected parents name: [A1]            <- this is user input
        > Expected children names: [A1#a, A1#b]. <- this is already in the class
    """
    def wrapper(instance, feat_x, feat_y, **kwargs):

        # Get all children onehot features from their parent names
        feat_x = [col for col in instance.cols if col.split('#')[0] in feat_x]
        feat_y = [col for col in instance.cols if col.split('#')[0] in feat_y]

        # Proceed as usual with the children instead of the parents.
        value = method(instance, feat_x, feat_y, **kwargs)
        return value
    return wrapper
