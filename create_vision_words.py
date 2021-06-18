import pickle

import numpy as np
from sklearn.cluster import KMeans


def multi_level_clustering(X=None, level=0, max_level=2, child_idx=0, n_clusters=8):
    print('level', level, 'child_idx', child_idx)
    kmeans = KMeans(n_clusters=n_clusters, verbose=True)
    kmeans.fit(X)
    model = {
        'model': kmeans
    }
    if (level < max_level):
        model['submodel'] = []
        cluster_idx = kmeans.predict(X)
        for idx in range(n_clusters):
            Xs = X[cluster_idx==idx]
            model['submodel'].append(multi_level_clustering(X=Xs, level=level+1, child_idx=idx))
    return model

def create_vision_words(list_sift_path='./list_sift.pickle', list_word_path='./words_model.pickle'):
    with open(list_sift_path, 'rb') as f:
        list_sift = pickle.load(f)
    list_sift = np.array(list_sift)
    list_sift /= 255.0
    model = multi_level_clustering(X=list_sift)
    with open(list_word_path, 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    create_vision_words()