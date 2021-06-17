from pymongo import MongoClient


class ImageRetrievalConnector:
    def __init__(self, *args, **kwargs) -> None:
        pass

class MongoDBConnector(ImageRetrievalConnector, MongoClient):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pass
