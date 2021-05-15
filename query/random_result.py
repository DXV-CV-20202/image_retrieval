from .query import Query

class RandomResult(Query):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'feature_size' in kwargs:
            self.output_size = kwargs['feature_size']
        else:
            self.output_size = (16,)

    def search(self, feature, k, *args, **kwargs):
        return [str(i) + '.png' for i in range(k)]