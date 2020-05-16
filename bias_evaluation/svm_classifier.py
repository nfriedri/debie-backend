import logging
import random
from sklearn import svm


def eval_svm(augmentations1, augmentations2, target1, target2, vocab, vecs):
    logging.info('Eval-Engine: SVM Classifier started')
    X_train = []
    y_train = []
    train = [(w, 0) for w in augmentations1 if w in vocab] + [(w, 1) for w in augmentations2 if w in vocab]

    scores = []
    for i in range(20):
        random.shuffle(train)

        for p in train:
            w = p[0]
            l = p[1]
            X_train.append(vecs[vocab[w]])
            y_train.append(l)

        # training SVM (rbf kernel)
        clf = svm.SVC(gamma='scale')
        clf.fit(X_train, y_train)

        # prediction data preparation
        X_test = []
        y_test = []
        test = [(w, 0) for w in target1 if w in vocab] + [(w, 1) for w in target2 if w in vocab]
        # print("SVM test words: " + str(len(test)))
        for p in test:
            w = p[0]
            l = p[1]
            X_test.append(vecs[vocab[w]])
            y_test.append(l)

        preds = clf.predict(X_test)
        correct = [i for i in range(len(y_test)) if y_test[i] == preds[i]]
        acc = len(correct) / len(preds)
        scores.append(acc)
    logging.info('Eval-Engine: SVM Classifier finished')
    return sum(scores) / len(scores)
