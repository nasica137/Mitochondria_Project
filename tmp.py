from monai.transforms.transform import Transform

class ApplyPreprocessing(Transform):
    def __init__(self, preprocess_fn, keys):
        """
        Args:
            preprocess_fn (callable): Preprocessing function to be applied to the specified keys.
            keys (str or list of str): Keys to be transformed.
        """
        self.preprocess_fn = preprocess_fn
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            data[key] = self.preprocess_fn(data[key])
        return data