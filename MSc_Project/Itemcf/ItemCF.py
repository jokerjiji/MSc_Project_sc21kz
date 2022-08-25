import math
import json
import pandas as pd
from pandas._libs.internals import defaultdict
from operator import itemgetter

from Evaluation.ItemCF_Evaluation_Matrics import ItemCF_Sample_Test
from Itemcf.Data_Preprocessing_test import LoadKKBoxData, SepKKBoxData


class ItemCF(object):

    def __init__(self, train_data, test_data, similarity='cosine', norm=True):
        self._train_data = train_data
        self._test_data = test_data
        self._similarity = similarity
        self._is_norm = norm
        self._co_occurrence_matrix = dict()
        self._item_simi_matrix = dict()

    def ItemCoOccurrenceMatrix(self):
        """
        Establish Item Co-occurrence Matrix
        :param train_data: User-Item Lookup Table
        :param similarity: Similarity calculation function selection
        :return: 
        """
        print('Item co-occurrence matrix generating...')
        N = defaultdict(int)  # Record the number of likes for each music
        for user, items in self._train_data.items():
            for i in items:
                self._co_occurrence_matrix.setdefault(i, dict())
                N[i] += 1
                for j in items:
                    if i == j:
                        continue
                    self._co_occurrence_matrix[i].setdefault(j, 0)
                    if self._similarity == "cosine":
                        self._co_occurrence_matrix[i][j] += 1
                    elif self._similarity == "iuf":
                        self._co_occurrence_matrix[i][j] += 1. / math.log1p(len(items) * 1.)

        occurrence_json = json.dumps(self._co_occurrence_matrix, sort_keys=False, indent=4, separators=(',', ': '))
        N_json = json.dumps(N, sort_keys=False, indent=4, separators=(',', ': '))
        # print(type(occurrence_json))
        f0 = open('Results/item_co-occurrence_matrix.json', 'w')
        f1 = open('Results/N.json', 'w')
        f0.write(occurrence_json)
        f1.write(N_json)
        print('Item co-occurrence matrix generated.')
        print('=======================================')
        return N

    def ItemSimilarityMatrix(self, N):
        """
        Calculating music similarity
        :param _co_occurrence_matrix:
        :param N: Record the number of likes for each music
        :param _is_norm:
        :return:
        """
        print('Item similarity matrix generating...')
        for i, related_items in self._co_occurrence_matrix.items():
            for j, cij in related_items.items():
                # Calculate similarity
                self._item_simi_matrix[i][j] = cij / math.sqrt(N[i] * N[j])

        # Normalize the item similarity matrix
        if self._is_norm:
            for i, relations in self._item_simi_matrix.items():
                if relations:
                    max_num = relations[max(relations, key=relations.get)]
                    # Return a new dict after normalization
                    self._item_simi_matrix[i] = {k: v / max_num for k, v in relations.items()}

        item_simi_json = json.dumps(self._item_simi_matrix, sort_keys=False, indent=4, separators=(',', ': '))
        f2 = open('Results/item_simi.json', 'w')
        f2.write(item_simi_json)
        print('Item similarity matrix generated.')
        print('=======================================')

    def Recommend(self, user, N, K):
        """
        :param trainData: User-Item表
        :param itemSimMatrix: 物品相似度矩阵
        :param user: 被推荐的用户user
        :param N: 推荐的商品个数
        :param K: 对每个用户喜爱物品在物品相似矩阵中找到与其最相似的K个
        :return: 按照user对推荐物品的感兴趣程度排序的N个商品
        """

        recommends = dict()
        items = self._test_data[user]
        # item_recommend_length = len(items) - 10
        # print('item_recommend_length: ' + str(item_recommend_length))
        items = items[:10]
        for idx, item in enumerate(items):
            # print(str(idx) + ': ' + )
            if item in self._item_simi_matrix:
                for i, sim in sorted(self._item_simi_matrix[item].items(), key=itemgetter(1), reverse=True)[:K]:
                    if i in items:
                        continue
                    recommends.setdefault(i, 0.)
                    recommends[i] += sim
        return dict(sorted(recommends.items(), key=itemgetter(1), reverse=True)[:N])

    def training(self):
        N = self.ItemCoOccurrenceMatrix()
        self.ItemSimilarityMatrix(N)


if __name__ == "__main__":
    # training ready 2022/8/14 03:10
    print('Data set loading...')
    train_filtered_df = pd.read_csv('../ProcessedData/merged_train_df_final.csv', low_memory=False)
    f1 = open('../ProcessedData/ItemCFData/train_set.json', 'r')
    f2 = open('../ProcessedData/ItemCFData/valid_set.json', 'r')
    f3 = open('../ProcessedData/ItemCFData/test_set.json', 'r')
    train_set = json.load(f1)
    valid_set = json.load(f2)
    test_set = json.load(f3)
    print("Train set user size: %d, valid set user size: %d, test set user size: %d" % (
        len(train_set), len(valid_set), len(test_set)))
    print('Data set loaded.')
    ItemCF = ItemCF(train_set, test_set, similarity='cosine', norm=True)
    print('Item similarity matrix loading...')
    f = open('Results/item_simi.json', 'r')
    item_simi = json.load(f)
    print('Item similarity matrix loaded.')
    ItemCF._item_simi_matrix = item_simi

    # training not ready
    # train_filtered_df = LoadKKBoxData()
    # train_set, valid_set, test_set = SepKKBoxData(train_filtered_df, 0.8)
    # print('Data pre-processed.')
    # print('=======================================')
    # train_set_json = json.dumps(train_set, sort_keys=False, indent=4, separators=(',', ': '))
    # valid_set_json = json.dumps(valid_set, sort_keys =False, indent=4, separators=(',', ': '))
    # test_set_json = json.dumps(test_set, sort_keys=False, indent=4, separators=(',', ': '))
    # f1 = open('../ProcessedData/ItemCFData/train_set.json', 'w')
    # f1.write(train_set_json)
    # f2 = open('../ProcessedData/ItemCFData/valid_set.json', 'w')
    # f2.write(valid_set_json)
    # f3 = open('../ProcessedData/ItemCFData/test_set.json', 'w')
    # f3.write(test_set_json)
    # print("Train set user size: %d, valid set user size: %d, test set user size: %d" % (
    #     len(train_set), len(valid_set), len(test_set)))
    # ItemCF = ItemCF(train_set, test_set, similarity='cosine', norm=True)
    # ItemCF.training()

    # 样例测试 sample test
    user_id = 'HlG3TJ4yvzo2A2xVQEtukQfGYM34sO1FiPCkNrEXIqk='
    ItemCF_Sample_Test(user_id, train_set, valid_set, test_set, ItemCF, train_filtered_df)

    # Precision/Recall/F1-Score
    # precision, recall, f1_score = Precision_Recall_Fscore(test_set, ItemCF)

    # Hit Ratio in Top N
    # HR_1, HR_5, HR_10, HR_20 = Hit_In_TopK(test_set, ItemCF)
