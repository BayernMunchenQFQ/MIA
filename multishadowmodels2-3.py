# 基于多攻击模型的成员推理攻击方法
# 影子模型之间参数设置不一样
# 目标模型和影子模型为SVM模型
# 将评估得到的攻击模型的攻击成功率乘上1000扩大不同的攻击模型之间权重的差异
# -*- coding: utf-8 -*-
from Config import config
from utils import *
import math

from sklearn.model_selection import train_test_split
import sklearn
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import random


def value(x, bias=0.1, base=math.e):
    # return 1/(-math.log(x, base) + bias)
    return math.exp(x*1000)


if __name__ == '__main__':
    print(config.train_data_path)
    for thenumberofexperiment in range(2):
        # N是影子模型的数量
        N = 5
        # MAX_EPOCH = random.random()
        shadow_C = random.randint(1, 10)
        # print(f'MAX_EPOCH:      {MAX_EPOCH}')
        LRNs = [random.randint(1, 10) for i in range(N)]
        Shadow_Cs = [random.randint(1, 10) for i in range(N)]
        bias = 0.1
        reports = []
        print(
            "####################################################################################################################")
        print()
        print(
            "####################################################################################################################")
        ####################################################################################################################

        ####################################################################################################################
        print('load data...')
        # 机器学习模型
        print("load target model data")
        target_train_x, target_train_y = load_data(config.train_data_path, config.target_train_num)  # 加载目标模型的数据和标签
        target_val_x, target_val_y = load_data(config.val_data_path, config.target_train_num)
        target_test_x, target_test_y = load_data(config.test_data_path, config.target_train_num)
        ####################################################################################################################

        ####################################################################################################################
        # print("load shadow models data")
        shadow_train_list = []
        shadow_val_list = []
        shadow_test_list = []
        # 加载N+1个影子模型的数据
        for i in range(N + 1):
            shadow_train_x, shadow_train_y = load_data(config.shadow_train_data_path,
                                                       config.shadow_train_num)  # 加载影子模型训练集数据
            shadow_train_list.append([shadow_train_x, shadow_train_y])  # 将不同的组对应的数据（数据和标签）存放在一个列表中
            shadow_val_x, shadow_val_y = load_data(config.shadow_val_data_path, config.shadow_train_num)
            shadow_val_list.append([shadow_val_x, shadow_val_y])
            shadow_test_x, shadow_test_y = load_data(config.shadow_test_data_path, config.shadow_train_num)
            shadow_test_list.append([shadow_test_x, shadow_test_y])
        ####################################################################################################################

        ####################################################################################################################
        # 目标模型数据tf-idf转换
        vectorizer = TfidfVectorizer()
        print('transform target model data...')
        target_train_x = vectorizer.fit_transform(target_train_x)
        target_test_x = vectorizer.transform(target_test_x)
        target_val_x = vectorizer.transform(target_val_x)
        ####################################################################################################################

        ####################################################################################################################
        print('train target model...')
        # 训练机器学习目标模型
        target_model = svm.SVC(C=config.C, kernel='rbf', probability=True, tol=5 * 1e-4)  # SVM模型
        target_model.fit(target_train_x, target_train_y)
        print("train score:", target_model.score(target_train_x, target_train_y))  # 训练集分类效果
        print("val score:", target_model.score(target_val_x, target_val_y))  # 验证集分类效果
        ####################################################################################################################

        ####################################################################################################################
        # 对目标模型的训练数据和测试集数据的一部分进行查询，并将目标模型的预测向量和其对应标签的（是否为成员）按一样的顺序存在attack_test_x和attak_test_y中,作为攻击模型的测试数据
        # 获取攻击模型测试数据
        print('get test attack data...')
        attack_test_x = []
        attack_test_y = []
        x = target_model.predict_proba(target_train_x[:config.attack_test_num])
        y = [1] * x.shape[0]
        x = x.tolist()
        attack_test_x += x
        attack_test_y += y
        x = target_model.predict_proba(target_test_x[:config.attack_test_num])
        y = [0] * x.shape[0]
        x = x.tolist()
        attack_test_x += x
        attack_test_y += y
        print(f"length of attack_test_x:{len(attack_test_x)}")
        print(
            "###########################################################################################################")
        print()
        print(
            "###########################################################################################################")
        ####################################################################################################################

        ####################################################################################################################
        models_list = []
        membership_x_proba_list = []
        none_membership_x_proba_list = []
        attack_test_x_proba_list = []
        attack_model_scores = []
        single_attack_results = []
        single_attack_reports = []
        for i in range(1, N + 1):
            # 形成1到N号影子模型的数据集
            attack_train_x = []
            attack_train_y = []
            # 影子模型的训练集中的包含对应序号的数据和0序号的数据
            shadow_train_x = shadow_train_list[i][0] + shadow_train_list[0][0]
            shadow_test_x = shadow_test_list[i][0] + shadow_test_list[0][0]
            shadow_val_x = shadow_val_list[i][0] + shadow_val_list[0][0]
            shadow_train_y = shadow_train_list[i][1] + shadow_train_list[0][1]
            shadow_test_y = shadow_test_list[i][1] + shadow_test_list[0][1]
            shadow_val_y = shadow_val_list[i][1] + shadow_val_list[0][1]

            print(f'transform {i} shadow models data...')
            # 对影子模型数据进行tf-idf转换
            shadow_train_x = vectorizer.fit_transform(shadow_train_x)
            shadow_test_x = vectorizer.transform(shadow_test_x)
            shadow_val_x = vectorizer.transform(shadow_val_x)

            # 影子模型训练集中对应序号的那部分数据，用于攻击模型的训练数据
            attack_membership_x = vectorizer.transform(shadow_train_list[i][0])
            attack_none_membership_x = vectorizer.transform(shadow_test_list[i][0])

            # 所有影子模型训练集中共有的数据，要用作线性预判器的训练数据
            membership_x = vectorizer.transform(shadow_train_list[0][0][:config.LM_train_num])
            # 所有影子模型训练集中都没有的数据，要用作线性预判器的训练数据
            none_membership_x = vectorizer.transform(shadow_test_list[0][0][:config.LM_train_num])
            ####################################################################################################################

            ####################################################################################################################
            # 机器学习影子模型
            # 训练影子模型
            print(f"train {i} shadow model...")

            print(f"LRN:             {LRNs[i-1]}")
            print(f"Shadow_C:         {Shadow_Cs[i-1]}")
            shadow_model = svm.SVC(C=Shadow_Cs[i-1], kernel="rbf", probability=True, tol=LRNs[i-1] * 1e-4)
            # shadow_model = svm.SVC(C=config.C, kernel='rbf', probability=True, tol=5 * 1e-4)
            shadow_model.fit(shadow_train_x, shadow_train_y)
            # 影子模型训练集分类效果
            print("train score:", shadow_model.score(shadow_train_x, shadow_train_y))
            # 影子模型验证集分类效果
            print("val score:", shadow_model.score(shadow_val_x, shadow_val_y))
            ####################################################################################################################

            ####################################################################################################################
            # 获得攻击模型训练集
            # print("get attack train data...")
            x = shadow_model.predict_proba(attack_membership_x[:config.attack_train_num])
            y = [1] * x.shape[0]
            x = x.tolist()
            attack_train_x += x
            attack_train_y += y

            x = shadow_model.predict_proba(attack_none_membership_x[:config.attack_train_num])
            y = [0] * x.shape[0]
            x = x.tolist()
            attack_train_x += x
            attack_train_y += y
            print(f"length of attack_train_x:{len(attack_train_x)}")
            ####################################################################################################################

            ####################################################################################################################
            # 获得公共攻击模型测试集
            print("get public attack val data...")
            membership_x = shadow_model.predict_proba(membership_x)
            membership_x = membership_x.tolist()
            membership_y = [1] * len(membership_x)

            none_membership_x = shadow_model.predict_proba(none_membership_x)
            none_membership_x = none_membership_x.tolist()
            none_membership_y = [0] * len(none_membership_x)

            LM_train_x = membership_x + none_membership_x
            LM_train_y = membership_y + none_membership_y
            ####################################################################################################################

            ####################################################################################################################
            # 训练机器学习攻击模型
            print(f'train {i} attack model...')
            attack_model = RandomForestClassifier(n_estimators=config.num_trees, random_state=11)
            attack_model.fit(attack_train_x, attack_train_y)
            # 得到单个攻击模型对目标模型的攻击效果
            single_attack_result = attack_model.score(attack_test_x, attack_test_y)
            # print(f"the attack result of the single attack model : {single_attack_result}")
            single_attack_results.append(single_attack_result)
            # print(classification_report(attack_test_y, attack_model.predict(attack_test_x)))
            rp = classification_report(attack_test_y, attack_model.predict(attack_test_x))
            single_attack_reports.append(rp)
            ####################################################################################################################

            ####################################################################################################################
            # 得到攻击模型对应公共测试集的分类效果
            # attack_model_score = attack_model.score(attack_test_x, attack_test_y)
            attack_model_score = attack_model.score(LM_train_x, LM_train_y)
            # print(f"the score of the attack model : {attack_model_score}")
            attack_model_scores.append(attack_model_score)  # 记录每个攻击模型在公共标准下的攻击效果
            # none_membership_x = attack_model.predict_proba(none_membership_x)
            # membership_x = attack_model.predict_proba(membership_x)
            # 获取攻击模型对目标模型生成的攻击模型测试数据的分类向量（2维）
            attack_test_x_proba = attack_model.predict_proba(attack_test_x)

            """""
            membership_x = [p[1] for p in membership_x]
            membership_x_proba_list.append(membership_x)

            none_membership_x = [p[1] for p in none_membership_x]
            none_membership_x_proba_list.append(none_membership_x)
            """""

            # 将攻击模型对目标模型产生的测试数据进行分类，并将对应的分类分数以列表形式存储起来。
            # 获取攻击模型判断测试集中每一个数据为目标模型成员数据的分数
            attack_test_x_proba = [p[1] for p in attack_test_x_proba]
            # 将每个攻击模型形成的上述列表以列表形式存储形成二维列表
            attack_test_x_proba_list.append(attack_test_x_proba)

            # 将影子模型和攻击模型以列表的形式存储起来
            models_list.append([shadow_model, attack_model])
            # print()
            # print()
        ####################################################################################################################

        ####################################################################################################################
        print(
            "###########################################################################################################")
        print()
        print(
            "###########################################################################################################")
        # print("get LM data...")
        """""
        LM_train_x = []
        LM_train_y = []
        none_membership_x = np.array(none_membership_x_proba_list)
        none_membership_x = none_membership_x.transpose()
        LM_train_x += none_membership_x.tolist()
        LM_train_y += [0] * len(none_membership_x)

        membership_x = np.array(membership_x_proba_list)
        membership_x = membership_x.transpose()
        LM_train_x += membership_x.tolist()
        LM_train_y += [1] * len(membership_x)
        """""
        # 将目标模型对攻击模型的测试数据转换成为numpy形式
        attack_test_x = np.array(attack_test_x_proba_list)
        attack_test_x = attack_test_x.transpose()
        LM_test_x = attack_test_x.tolist()
        LM_test_y = attack_test_y

        # print("train LM...")
        # 对不同的攻击模型对公共数据的攻击效果进行softmax计算形成新的分数列表
        min_score = min(attack_model_scores)
        attack_model_scores2 = [x-min_score for x in attack_model_scores]
        score_sum = 0
        for item in attack_model_scores2:
            score_sum += value(item)
        attack_model_weights = [value(score) / score_sum for score in attack_model_scores2]
        ####################################################################################################################

        ####################################################################################################################
        print("membership inference attack...")
        # 将不同的攻击模型对同一条数据的分类分数进行加权平均计算得到最后的分类分数，已进行最终分类
        for k in range(len(LM_test_x)):
            score = 0
            for h in range(N):
                score += attack_model_weights[h] * LM_test_x[k][h]
            LM_test_x[k] = score
        print(LM_test_x)
        test_result = [1 if p > 0.5 else 0 for p in LM_test_x]
        # 计算并输出最终的分类结果
        test_right_num = 0
        TP = 0
        FP = 0
        FN = 0
        for j in range(len(test_result)):
            if (test_result[j] == LM_test_y[j]):
                test_right_num += 1
            if (test_result[j] == LM_test_y[j]) and test_result[j] == 1:
                TP += 1
            if (test_result[j] != LM_test_y[j]) and test_result[j] == 1:
                FP += 1
            if (test_result[j] != LM_test_y[j]) and test_result[j] == 0:
                FN += 1
        multi_attack_accuracy = test_right_num / len(test_result)
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1 = 2 * Precision * Recall / (Precision + Recall)
        print(f"test_num : {len(test_result)}")
        print(f"multi_attack_accuracy : {multi_attack_accuracy}")
        print(f"multi_attack_precision : {Precision}")
        print(f"multi_attack_recall : {Recall}")
        print(f"multi_attack_f1 : {F1}")
        print(
            "-----------------------------------------------------------------------------------------------------------------")
        for i in range(N):
            print(f"the {i + 1} attack model : ")
            print(f"attack accuracy {single_attack_results[i]}")
            print(f"score {attack_model_weights[i]}")
            print(f"attack score  {attack_model_scores[i]}")
            print(single_attack_reports[i])
            print()
            print(
                "-----------------------------------------------------------------------------------------------------------------")
            print()
    """""
        LM = LinearRegression()
        LM_data_list = list(zip(LM_train_x, LM_train_y))
        shuffle(LM_data_list)
        LM_train_x[:], LM_train_y[:] = zip(*LM_data_list)
        LM.fit(LM_train_x, LM_train_y)
        train_proba = LM.predict(LM_train_x)
        print(LM_test_x)
        train_result = [1 if p > 0.5 else 0 for p in train_proba ]
        train_right_num = 0
        for j in range(len(train_result)):
            if(train_result[j] == LM_train_y[j]):
                train_right_num += 1
        train_accuracy = train_right_num / len(train_result)
        print(f"train_num : {len(train_result)}")
        print(f"train_accuracy : {train_accuracy}")
        print("train LM score:", LM.score(LM_train_x, LM_train_y))

        test_proba = LM.predict(LM_test_x)
        test_result = [1 if p > 0.5 else 0 for p in test_proba ]
        test_right_num = 0
        for j in range(len(test_result)):
            if(test_result[j] == LM_test_y[j]):
                test_right_num += 1
        test_accuracy = test_right_num / len(test_result)
        print(f"test_num : {len(test_result)}")
        print(f"test_accuracy : {test_accuracy}")
        print("test LM score:", LM.score(LM_test_x, LM_test_y))
        """""
