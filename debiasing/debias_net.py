import numpy
import calculation
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVR

# In DEVELOPMENT

def debias_net(target_set1, target_set2, argument_set, lambda_value=0.2):
    target1, target2, argument = calculation.create_duplicates(target_set1, target_set2, argument_set)
    t1_list, t2_list, arg_list = calculation.transform_multiple_dicts_to_lists(target1, target2, argument)


def learn_debiasnet_algorithm(t1_list, t2_list, a_list):
    estimator = LinearSVR()
    value_list = []
    i = -1
    while i != 1:
        value_list.append(i)
        i += 0.0001
    print('Value_list: ' + str(len(value_list)))
    grid_param = {'value': value_list}
    grid_search = GridSearchCV(estimator, grid_param, cv=5, n_jobs=-1)
    grid_search.fit(t1_list)


def loss_function_ld(t1_list, t2_list, a_list):
    l_d = []
    for i in range(len(t1_list)):
        for j in range(len(t2_list)):
            for k in range(len(a_list)):
                value = (calculation.cosine_similarity(t1_list[i], a_list[k]) - calculation.cosine_similarity(
                    t2_list[j], a_list[k])) ^ 2
                l_d.append(value)