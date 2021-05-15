from feature_extraction.random_feature_extraction import RandomFeatureExtraction
from query.random_result import RandomResult

kwargs = {
    'feature_size': (5, 5)
}

rfe = RandomFeatureExtraction(**kwargs)
ft = rfe.extract_feature(None)
print(ft)

rr = RandomResult(**kwargs)
img = rr.search(ft, 2)
print(img)