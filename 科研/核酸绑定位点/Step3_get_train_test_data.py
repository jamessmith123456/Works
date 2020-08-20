import pickle
import numpy as np
import random
import copy

def binary(data, threshold = 30):
    data[ data < threshold ] = 0
    data[ data >= threshold ] = 1
    pos_num = 0
    neg_num = 0
    for i in range(data.shape[0]):
        if data[i] == 1:
            pos_num += 1
        if data[i] == 0:
            neg_num += 1
    print("pos_example_num:{},neg_example_num:{}".format(pos_num, neg_num))
    return data

def split_train_test(label):
    #分成五折交叉验证
    pos_index = []
    neg_index = []
    for i in range(label.shape[0]):
        if label[i]==1:
            pos_index.append(i)
        else:
            neg_index.append(i)
    random.shuffle(pos_index)
    random.shuffle(neg_index)
    pos_interval = (len(pos_index))//5
    neg_interval = (len(neg_index))//5

    test_pos_index1 = pos_index[0:pos_interval]
    test_neg_index1 = neg_index[0:neg_interval]
    test_index1 = copy.deepcopy(test_pos_index1)
    test_index1.extend(test_neg_index1)
    train_pos_index1 = list(set(pos_index)-set(test_pos_index1))
    train_neg_index1 = list(set(neg_index)-set(test_neg_index1))
    train_index1 = copy.deepcopy(train_pos_index1)
    train_index1.extend(train_neg_index1)
    random.shuffle(test_index1)
    random.shuffle(train_index1)

    test_pos_index2 = pos_index[1*pos_interval:2*pos_interval]
    test_neg_index2 = neg_index[1*neg_interval:2*neg_interval]
    test_index2 = copy.deepcopy(test_pos_index2)
    test_index2.extend(test_neg_index2)
    train_pos_index2 = list(set(pos_index)-set(test_pos_index2))
    train_neg_index2 = list(set(neg_index)-set(test_neg_index2))
    train_index2 = copy.deepcopy(train_pos_index2)
    train_index2.extend(train_neg_index2)
    random.shuffle(test_index2)
    random.shuffle(train_index2)

    test_pos_index3 = pos_index[2*pos_interval:3*pos_interval]
    test_neg_index3 = neg_index[2*neg_interval:3*neg_interval]
    test_index3 = copy.deepcopy(test_pos_index3)
    test_index3.extend(test_neg_index3)
    train_pos_index3 = list(set(pos_index)-set(test_pos_index3))
    train_neg_index3 = list(set(neg_index)-set(test_neg_index3))
    train_index3 = copy.deepcopy(train_pos_index3)
    train_index3.extend(train_neg_index3)
    random.shuffle(test_index3)
    random.shuffle(train_index3)

    test_pos_index4 = pos_index[3*pos_interval:4*pos_interval]
    test_neg_index4 = neg_index[3*neg_interval:4*neg_interval]
    test_index4 = copy.deepcopy(test_pos_index4)
    test_index4.extend(test_neg_index4)
    train_pos_index4 = list(set(pos_index)-set(test_pos_index4))
    train_neg_index4 = list(set(neg_index)-set(test_neg_index4))
    train_index4 = copy.deepcopy(train_pos_index4)
    train_index4.extend(train_neg_index4)
    random.shuffle(test_index4)
    random.shuffle(train_index4)

    test_pos_index5 = pos_index[4*pos_interval:]
    test_neg_index5 = neg_index[4*neg_interval:]
    test_index5 = copy.deepcopy(test_pos_index5)
    test_index5.extend(test_neg_index5)
    train_pos_index5 = list(set(pos_index)-set(test_pos_index5))
    train_neg_index5 = list(set(neg_index)-set(test_neg_index5))
    train_index5 = copy.deepcopy(train_pos_index5)
    train_index5.extend(train_neg_index5)
    random.shuffle(test_index5)
    random.shuffle(train_index5)

    All_index = [[train_index1, test_index1], [train_index2, test_index2], [train_index3, test_index3], [train_index4, test_index4], [train_index5, test_index5]]
    return All_index

def examine_split_right(All_index, all_label):
    raw_index = list(range(len(All_index[0][0])+len(All_index[0][1])))
    for i in range(len(All_index)):
        temp_train_index = copy.deepcopy(All_index[i][0])
        temp_test_index = copy.deepcopy(All_index[i][1])
        temp_train_pos_num = 0
        temp_train_neg_num = 0
        temp_test_pos_num = 0
        temp_test_neg_num = 0
        for item in temp_train_index:
            if all_label[item]==0:
                temp_train_neg_num += 1
            else:
                temp_train_pos_num += 1
        for item in temp_test_index:
            if all_label[item]==0:
                temp_test_neg_num += 1
            else:
                temp_test_pos_num += 1
        print("fold:{}_trainposnum:{}".format(i,temp_train_pos_num))
        print("fold:{}_trainnegnum:{}".format(i, temp_train_neg_num))
        print("fold:{}_testposnum:{}".format(i,temp_test_pos_num))
        print("fold:{}_testnegnum:{}".format(i, temp_test_neg_num))

        temp_train_index.extend(temp_test_index)
        temp_train_index = list(set(temp_train_index))
        if temp_train_index==raw_index:
            print("right")
        else:
            print("wrong")


# def split_train_test(label, ratio):
#     pos_index = []
#     neg_index = []
#     for i in range(label.shape[0]):
#         if label[i]==1:
#             pos_index.append(i)
#         else:
#             neg_index.append(i)
#     random.shuffle(pos_index)
#     random.shuffle(neg_index)
#     train_index = []
#     test_index = []
#     for i in range(len(pos_index)):
#         if i<=int(len(pos_index)*ratio):
#             test_index.append(pos_index[i])
#         else:
#             train_index.append(pos_index[i])
#     for i in range(len(neg_index)):
#         if i<=int(len(neg_index)*ratio):
#             test_index.append(neg_index[i])
#         else:
#             train_index.append(neg_index[i])
#     random.shuffle(test_index)
#     random.shuffle(train_index)
#
#     return train_index, test_index
if __name__ == '__main__':
    #load_file = open("F:/DrugScreening2/mydata2/targets_drug/feature_pkl/cellline_pssm_pss_feature.pkl",'rb')
    load_file = open("./cellline_pssm_pss_feature.pkl", 'rb')
    all_cellline = pickle.load(load_file)  #21种疾病
    all_cellline_feature = pickle.load(load_file) #每个疾病是键  键值是(115,1279)的array
    load_file.close()

    #load_file = open("morgan_feature.pkl","rb")
    load_file = open("./morgan_feature.pkl", "rb")
    all_drug = pickle.load(load_file) #35种药物  每种药物的键值是它的SMILE分子格式
    all_drug_MACC = pickle.load(load_file) #MACC是长为167的0-1向量   33种药物
    all_drug_Morgan = pickle.load(load_file) #Morgan是长为1024的0-1向量 33种药物
    all_feature_drug = pickle.load(load_file) #33种药物的名字
    score = pickle.load(load_file)
    load_file.close()

    #with open('F:/DrugScreening2/mydata2/raw_data/drug_drug_synergy_original.csv','r') as f:
    with open('./drug_drug_synergy_original.csv', 'r') as f:
        content = f.readlines()
    content = content[1:]

    all_sample = []
    all_label = []
    all_feature1 = []
    all_feature2 = []
    all_feature3 = []

    count = 0
    for i in range(len(content)):
        temp = content[i].split(',')
        if temp[1] in all_drug_Morgan.keys() and temp[2] in all_drug_Morgan.keys() and temp[3] in all_cellline:
            count += 1 #一共9303个
            temp_sample = [temp[1], temp[2], temp[3], float(temp[4])]
            all_sample.append(temp_sample) #一共9303个元素 每个元素是一个为4的list  ['5-FU', 'ABT-888', 'A2058', 7.6935]
            all_label.append(float(temp[4]))
            all_feature1.append(all_drug_Morgan[temp[1]]) #9303个元素 每个元素为1024
            all_feature2.append(all_drug_Morgan[temp[2]]) ##9303个元素 每个元素为1024
            all_feature3.append(all_cellline_feature[temp[3]]) #9303个元素 每个元素又包含115个长为1279的list

    all_feature1 = np.array(all_feature1)
    all_feature2 = np.array(all_feature2)
    all_feature3 = np.array(all_feature3)

    all_label = np.array(all_label)
    all_label = binary(all_label, 30) #正样本880 负样本8423

    #train_index, test_index = split_train_test(all_label, 0.2) #训练样本7441  测试样本：1862
    # All_index = split_train_test(all_label)
    # all_feature1_train = all_feature1[train_index, :]
    # all_feature2_train = all_feature2[train_index, :]
    # all_feature3_train = all_feature3[train_index, :, :]
    #
    # all_feature1_test = all_feature1[test_index, :]
    # all_feature2_test = all_feature2[test_index, :]
    # all_feature3_test = all_feature3[test_index, :, :]
    #
    # train_label = all_label[train_index]
    # test_label = all_label[test_index]
    #
    # save_file = open('train_and_test.pkl', 'wb')
    # pickle.dump(all_feature1_train, save_file, protocol = 4) #(7441,1024)
    # pickle.dump(all_feature2_train, save_file, protocol = 4) #(7441,1024)
    # pickle.dump(all_feature3_train, save_file, protocol = 4) #(7441,115,1279)
    # pickle.dump(all_feature1_test, save_file, protocol = 4) #(1862,1024)
    # pickle.dump(all_feature2_test, save_file, protocol = 4) #(1862,1024)
    # pickle.dump(all_feature3_test, save_file, protocol = 4) #(1862,115,1279)
    # pickle.dump(train_label, save_file, protocol = 4) #(7441,)
    # pickle.dump(test_label, save_file, protocol = 4) #(1862,)
    # save_file.close()
    #
    # print("all_feature1_train.shape:", all_feature1_train.shape)
    # print("all_feature2_train.shape:", all_feature2_train.shape)
    # print("all_feature3_train.shape:", all_feature3_train.shape)
    # print("all_feature1_test.shape:", all_feature1_test.shape)
    # print("all_feature2_test.shape:", all_feature2_test.shape)
    # print("all_feature3_test.shape:", all_feature3_test.shape)
    # print("train_label.shape:", train_label.shape)
    # print("test_label.shape:", test_label.shape)
    # train_pos_num = 0
    # train_neg_num = 0
    # test_pos_num = 0
    # test_neg_num = 0
    # for i in range(train_label.shape[0]):
    #     if train_label[i]==0:
    #         train_neg_num += 1
    #     else:
    #         train_pos_num += 1
    # for i in range(test_label.shape[0]):
    #     if test_label[i]==0:
    #         test_neg_num += 1
    #     else:
    #         test_pos_num += 1
    # print("train_pos_num:{}, train_neg_num:{}".format(train_pos_num, train_neg_num))
    # print("test_pos_num:{}, test_neg_num:{}".format(test_pos_num, test_neg_num))

    All_index = split_train_test(all_label)
    examine_split_right(All_index, all_label)
    for i in range(len(All_index)):
        temp_train_index = All_index[i][0]
        temp_test_index = All_index[i][1]
        all_feature1_train = all_feature1[temp_train_index, :]
        all_feature2_train = all_feature2[temp_train_index, :]
        all_feature3_train = all_feature3[temp_train_index, :, :]

        all_feature1_test = all_feature1[temp_test_index, :]
        all_feature2_test = all_feature2[temp_test_index, :]
        all_feature3_test = all_feature3[temp_test_index, :, :]

        train_label = all_label[temp_train_index]
        test_label = all_label[temp_test_index]

        save_file = open('train_and_test'+str(i)+'.pkl', 'wb')
        pickle.dump(all_feature1_train, save_file, protocol = 4) #(7441,1024)
        pickle.dump(all_feature2_train, save_file, protocol = 4) #(7441,1024)
        pickle.dump(all_feature3_train, save_file, protocol = 4) #(7441,115,1279)
        pickle.dump(all_feature1_test, save_file, protocol = 4) #(1862,1024)
        pickle.dump(all_feature2_test, save_file, protocol = 4) #(1862,1024)
        pickle.dump(all_feature3_test, save_file, protocol = 4) #(1862,115,1279)
        pickle.dump(train_label, save_file, protocol = 4) #(7441,)
        pickle.dump(test_label, save_file, protocol = 4) #(1862,)
        save_file.close()

        print("all_feature1_train.shape:", all_feature1_train.shape)
        print("all_feature2_train.shape:", all_feature2_train.shape)
        print("all_feature3_train.shape:", all_feature3_train.shape)
        print("all_feature1_test.shape:", all_feature1_test.shape)
        print("all_feature2_test.shape:", all_feature2_test.shape)
        print("all_feature3_test.shape:", all_feature3_test.shape)
        print("train_label.shape:", train_label.shape)
        print("test_label.shape:", test_label.shape)
        train_pos_num = 0
        train_neg_num = 0
        test_pos_num = 0
        test_neg_num = 0
        for j in range(train_label.shape[0]):
            if train_label[j]==0:
                train_neg_num += 1
            else:
                train_pos_num += 1
            if train_neg_num==1:
                print("first train neg:",j)
            if train_pos_num==1:
                print("first train pos:",j)
        for j in range(test_label.shape[0]):
            if test_label[j]==0:
                test_neg_num += 1
            else:
                test_pos_num += 1
            if test_neg_num==1:
                print("first test neg:",j)
            if test_pos_num==1:
                print("first test pos:",j)
        print("train_pos_num:{}, train_neg_num:{}".format(train_pos_num, train_neg_num))
        print("test_pos_num:{}, test_neg_num:{}".format(test_pos_num, test_neg_num))

