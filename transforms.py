from monai.transforms import MapTransform  # Ensure MapTransform is importe

class CustomTransform(MapTransform):
    def __init__(self, keys, preprocess_input):
        super().__init__(keys)
        self.preprocess_input = preprocess_input
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                # Apply augmentations based on the provided keys
                img = d[key]
                #img = self.rand_flip(img)
                img = self.preprocess_input(img)
                # Update the augmented image in the dictionary
                d[key] = img
        return d