import monai


class PrintShaped(monai.transforms.InvertibleTransform):
    """
    Convenience MONAI transform that prints the shape of elements in a
    dictionary of tensors. Used for debugging.
    """

    def __init__(self, prefix=""):
        self.prefix = prefix

    def __call__(self, X):
        for k in X:
            try:
                print(self.prefix, k, X[k].shape)
            except Exception:
                pass
        return X

    def inverse(self, X):
        return self(X)


class PrintSumd(monai.transforms.InvertibleTransform):
    """
    Convenience MONAI transform that prints the sum of elements in a
    dictionary of tensors. Used for debugging.
    """

    def __init__(self, prefix=""):
        self.prefix = prefix

    def __call__(self, X):
        for k in X:
            try:
                print(self.prefix, k, X[k].sum())
            except Exception:
                pass
        return X

    def inverse(self, X):
        return self(X)


class PrintRanged(monai.transforms.InvertibleTransform):
    """
    Convenience MONAI transform that prints the sum of elements in a
    dictionary of tensors. Used for debugging.
    """

    def __init__(self, prefix="", keys: list[str] = None):
        self.prefix = prefix
        self.keys = keys

    def __call__(self, X):
        if self.keys is not None:
            keys = self.keys
        else:
            keys = X.keys()
        for k in keys:
            try:
                print(self.prefix, k, X[k].min(), X[k].max())
            except Exception:
                pass
        return X

    def inverse(self, X):
        return self(X)


class PrintTyped(monai.transforms.InvertibleTransform):
    """
    Convenience MONAI transform that prints the type of elements in a
    dictionary of tensors. Used for debugging.
    """

    def __init__(self, prefix=""):
        self.prefix = prefix

    def __call__(self, X):
        for k in X:
            print(self.prefix, k, type(X[k]))
        return X

    def inverse(self, X):
        return self(X)


class Printd(monai.transforms.InvertibleTransform):
    """
    Convenience MONAI transform that prints elements. Used for debugging.
    """

    def __init__(self, prefix="", keys=None):
        self.prefix = prefix
        self.keys = keys

    def __call__(self, X):
        for k in X:
            if self.keys is not None:
                if k in self.keys:
                    print(self.prefix, k, X[k])
            else:
                print(self.prefix, k, X[k])
        return X

    def inverse(self, X):
        return self(X)
