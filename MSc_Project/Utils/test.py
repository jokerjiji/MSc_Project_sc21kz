import numpy as np
import pandas as pd

# np.random.seed(3)
#
# for i in range(10):
#     for j in range(10):
#         a = np.random.randint(100)
#         print(a)





# dict1 = {}
#
# list1 = ['aa', 'bb']
# dict1['test'] = list1
# dict1['test'].append('cc')
# print(dict1)
# items = ['aabb', 'ccdd']
# train_data = dict()
# for item in items:
#     train_data.setdefault('user', [])
#     train_data['user'].append(item)
#
# print(train_data)

# for user, item in raw_data:
#     train_data.setdefault(user, list)
#     train_data[user].append(item)
# print(train_data)



# k = np.load("test_set.npy", allow_pickle=True)
# print(type(k))
# # print(k)

# feafile = 'Evaluation/fea_col.npy'
#
# fea_col = np.load(feafile, allow_pickle=True)
#
# print(fea_col)
#
# print('----------')
#
feafile = 'Utils/fea_col.npy'

fea_col = np.load(feafile, allow_pickle=True)

print(fea_col)
#
# # #
# print(fea_col[1])
# precision = 1
# recall = 0
#
#
# if ((precision == 0) & (recall == 0)):
#     print('111')
# else:
#     print('222')

# sparse_feas = ['city', 'gender', 'genre_ids', 'artist_name', 'composer', 'lyricist', 'language']
# fea_list_sparse = [{'feat': 'bd'}, {'feat': 'song_length'}]
# fea_list_dense = [{'feat': 'city', 'feat_num': , 'embed_dim': 8}, {'feat': 'gender', 'feat_num': , 'embed_dim': 8}, {'feat': 'genre_ids', 'feat_num': , 'embed_dim': 8}, {'feat': 'artist_name', 'feat_num': , 'embed_dim': 8}, {'feat': 'composer', 'feat_num': , 'embed_dim': 8}, {'feat': 'lyricist', 'feat_num': , 'embed_dim': 8}, {'feat': 'language', 'feat_num': , 'embed_dim': 8}]


# 生成fea_col.npy
# data_df = pd.read_csv('D:/PycharmProject/MSc_Project/merged_train_df_final.csv', low_memory=False)
# sparse_feas = ['city', 'gender', 'genre_ids', 'artist_name', 'composer', 'lyricist', 'language']
# fea_list_dense = [{'feat': 'bd'}, {'feat': 'song_length'}]
# fea_list_sparse = []
# for fea in sparse_feas:
#     dict = {}
#     num_distinct = data_df.drop_duplicates([fea]).shape[0]
#     print(fea + ': ' + str(num_distinct))
#     dict['feat'] = fea
#     dict['feat_num'] = num_distinct
#     dict['embed_dim'] = 8
#     fea_list_sparse.append(dict)
# print(fea_list_sparse)
# fea_list = [fea_list_dense, fea_list_sparse]
# fea_list = np.array(fea_list)
# np.save('Utils/fea_col.npy', fea_list)
#



