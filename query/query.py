
class Query:
    def __init__(self, *args, **kwargs):
        pass
    
    def search(self, feature, k, *args, **kwargs):
        raise Exception("search function must be implemented")