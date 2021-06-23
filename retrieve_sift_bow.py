from feature_extractor import SIFT
import pickle
import json
import time
import cv2
import numpy as np

def get_word_id(word_model, word):
    _id = word_model['model'].predict([word])[0]
    if 'id' in word_model:
        return _id + word_model['id'], 1.0 / np.linalg.norm(word-word_model['model'].cluster_centers_[_id])
    return get_word_id(word_model['submodel'][_id], word)


def main(
    word_model_path='words_model.pickle',
    inverted_file_path='sift_inverted_file.pickle',
    testset_path = './data/cifar-10/test.json'
):
    with open(word_model_path, 'rb') as f:
        word_model = pickle.load(f)
    count = {'id': 0}
    def assign_model_id(model, count):
        if ('submodel' not in model) or (len(model['submodel']) <= 0):
            model['id'] = count['id']
            count['id'] += model['model'].cluster_centers_.shape[0]
        else:
            for submodel in model['submodel']:
                assign_model_id(submodel, count)
    assign_model_id(word_model, count)

    with open(inverted_file_path, 'rb') as f:
        inverted_file = pickle.load(f)

    with open(testset_path) as f:
        testset_des = json.load(f)
    extractor = SIFT()

    all_classes = [
        'airplane', 'automobile', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse',
        'ship', 'truck'
    ]
    list_n_top = [1, 3, 5, 10]
    count_success = dict()
    count = dict()
    total = dict()
    confusion_matrix = dict()
    for n_top in list_n_top:
        count_success[n_top] = count[n_top] = total[n_top] = 0
        confusion_matrix[n_top] = dict()
        for c1 in all_classes:
            confusion_matrix[n_top][c1] = dict()
            for c2 in all_classes:
                confusion_matrix[n_top][c1][c2] = 0
    sample_count = 0

    start_time = time.time()
    for idx, des in enumerate(testset_des):
        image = cv2.imread(des['image_path'])
        _, words = extractor.extract_full(image)
        if type(words) == type(None):
            words = np.zeros((1, 128))
        words /= 255.0
        for word in words:
            sim_cnt = dict()
            _id, coef = get_word_id(word_model, word)
            for img in inverted_file[_id]:
                if img not in sim_cnt:
                    sim_cnt[img] = 0
                sim_cnt[img] += inverted_file[_id][img] * coef
        sim_lst = [(img, sim_cnt[img]) for img in sim_cnt]
        res = sorted(sim_lst, key=lambda x: x[1], reverse=True)

        # print([r[1] for r in res[:15]])

        image_class_name = des['class_name']
        for n_top in list_n_top:
            records_class_name = [r[0].split('/')[-2] for r in res[:n_top]]
            # print(records_class_name)
            for record_class_name in records_class_name:
                confusion_matrix[n_top][image_class_name][record_class_name] += 1
                if image_class_name == record_class_name:
                    count[n_top] += 1
            if image_class_name in records_class_name:
                count_success[n_top] += 1
            total[n_top] += len(records_class_name)

        sample_count += 1
        if sample_count % 100 == 0:
            msg = []
            msg2 = []
            for n_top in list_n_top:
                msg.append(' '.join(['%d:' % (n_top,), "%.2f" % (count[n_top] * 100 / total[n_top]) + '%']))
                msg2.append(' '.join(['%d:' % (n_top,), "%.2f" % (count_success[n_top] * 100 / sample_count) + '%']))
            print(sample_count, 'accuracy' + '; '.join(msg), 'success:' + '; '.join(msg2))
        # if idx == 400:
        #     break
    print('Time:', time.time() - start_time)
    for n_top in list_n_top:
        print('Top %d accuracy:' % (n_top,), str(count[n_top] * 100 / total[n_top]) + '%')
        print('Top %d success:' % (n_top,), str(count_success[n_top] * 100 / sample_count) + '%')

    n_top = 10
    print()
    for c2 in all_classes:
        print('%12s' % c2, end='')
    print()
    for c1 in all_classes:
        for c2 in all_classes:
            print('%12s' % int(confusion_matrix[n_top][c1][c2]), end='')
        print()
if __name__ == '__main__':
    main()