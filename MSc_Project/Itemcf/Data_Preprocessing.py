import pandas as pd
import random
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def LoadKKBoxData():
    print('Loading data...')
    train_file_path = "C:/Users/35402/OneDrive - University of Leeds/Msc Project/Datasets/kkbox-music-recommendation-challenge/train.csv"
    song_file_path = "C:/Users/35402/OneDrive - University of Leeds/Msc Project/Datasets/kkbox-music-recommendation-challenge/songs.csv"
    song_detail_file_path = "C:/Users/35402/OneDrive - University of Leeds/Msc Project/Datasets/kkbox-music-recommendation-challenge/song_extra_info.csv"
    user_file_path = "C:/Users/35402/OneDrive - University of Leeds/Msc Project/Datasets/kkbox-music-recommendation-challenge/members.csv"
    #user
    user_df = pd.read_csv(user_file_path)
    user_notnull = user_df.dropna()
    user_cleaned = user_notnull[(user_notnull['bd'] != 0)].reset_index(drop=True)

    #song
    song_df = pd.read_csv(song_file_path)
    song_df_notnull = song_df.dropna()

    #song detail
    song_detail_df = pd.read_csv(song_detail_file_path)
    song_detail_df = song_detail_df[['song_id', 'name']]
    merged_song_df = song_df_notnull.merge(song_detail_df, on=['song_id'])

    #train
    train_df = pd.read_csv(train_file_path)
    train_df = train_df[['msno', 'song_id', 'target']]
    train_df_notnull = train_df.dropna()
    train_df_positive = train_df_notnull[(train_df_notnull['target'] == 1)].reset_index(drop=True)
    merged_train_df = train_df_positive.merge(user_cleaned, on=['msno'])
    merged_train_df = merged_train_df.merge(merged_song_df, on=['song_id'])
    merged_train_df.rename(columns={'msno': 'user_id'}, inplace=True)

    user_record_dict = {}
    for i, row in merged_train_df.iterrows():
        if row['user_id'] not in user_record_dict:
            user_record_dict[row['user_id']] = 1
        else:
            user_record_dict[row['user_id']] += 1

    user_record_df = pd.DataFrame.from_dict(user_record_dict, orient='index', columns=['counts'])
    user_record_df = user_record_df.reset_index().rename(columns={'index': 'user_id', 'counts': 'user_counts'})
    user_record_df1 = user_record_df[(user_record_df['user_counts'] >= 20)].reset_index(drop=True)
    # user_record_df1 = user_record_df[(user_record_df['user_counts'] <= 400)].reset_index(drop=True)
    merged_train_df_1 = merged_train_df.merge(user_record_df1, on=['user_id'])

    song_record_dict = {}
    for i, row in merged_train_df_1.iterrows():
        if row['song_id'] not in song_record_dict:
            song_record_dict[row['song_id']] = 1
        else:
            song_record_dict[row['song_id']] += 1

    song_record_df = pd.DataFrame.from_dict(song_record_dict, orient='index', columns=['counts'])
    song_record_df = song_record_df.reset_index().rename(columns={'index': 'song_id', 'counts': 'song_counts'})
    song_record_df1 = song_record_df[(song_record_df['song_counts'] > 1)].reset_index(drop=True)
    merged_train_df_final = merged_train_df_1.merge(song_record_df1, on=['song_id'])

    print('number of dataset: ' + str(len(merged_train_df_final)))

    merged_train_df_final.to_csv('merged_train_df_final.csv', index=False)
    print('merged_train_df_final.csv generated.')
    print('Data loaded.')
    print('=======================================')
    return merged_train_df_final


def PreProcessData(raw_data, item_list, neg_ratio, tag):
    """
    Establish User-Item lookup table, following the below data structure：
        {"User1": {MusicID1, MusicID2, MusicID3,...}
         "User2": {MusicID12, MusicID5, MusicID8,...}
         ...
        }
    """
    print('Data pre-processing...')
    train_data = dict()
    for user, item in raw_data:
        train_data.setdefault(user, [])
        train_data[user].append(item)

    return train_data
    # # negative sampling
    # NegSampling(train_data, item_list, neg_ratio, tag)



def NegSampling(data_dict, item_list, neg_ratio, tag):
    # def get_neg_sample(self, word_index, array):
    if tag == 'test':
        file_path = 'ProcessedData/DeepData/' + tag + '_df.csv'
        data_list = []
        np.random.seed(3)
        max_len_user_history = 0
        for user in data_dict:
            len_user_history = len(data_dict[user][10:])
            neg_list = []
            if len_user_history > max_len_user_history:
                max_len_user_history = len_user_history
            for item in data_dict[user][10:]:
                data_list.append([user, item, 1])
                neg_count = 0
                while neg_count < neg_ratio:
                    neg_sample = item_list[np.random.randint(len(item_list))]
                    if neg_sample in data_dict[user][10:]:
                        continue
                    elif neg_sample in neg_list:
                        continue
                    else:
                        neg_list.append(neg_sample)
                        # data_dict[user].append(neg_sample)
                        data_list.append([user, neg_sample, 0])
                        neg_count += 1
    else:
        file_path = 'ProcessedData/DeepData/' + tag + '_df.csv'
        data_list = []
        np.random.seed(3)
        max_len_user_history = 0
        for user in data_dict:
            len_user_history = len(data_dict[user])
            neg_list = []
            if len_user_history > max_len_user_history:
                max_len_user_history = len_user_history
            for item in data_dict[user]:
                data_list.append([user, item, 1])
                neg_count = 0
                while neg_count < neg_ratio:
                    neg_sample = item_list[np.random.randint(len(item_list))]
                    if neg_sample in data_dict[user]:
                        continue
                    elif neg_sample in neg_list:
                        continue
                    else:
                        neg_list.append(neg_sample)
                        # data_dict[user].append(neg_sample)
                        data_list.append([user, neg_sample, 0])
                        neg_count += 1
    data_df = pd.DataFrame(data_list, columns=['user_id', 'song_id', 'label'])
    data_df.to_csv(file_path, index=False)
    print(tag + '_df.csv generated.')
    print('Max user history length in ' + tag + ' set is: ' + str(max_len_user_history))
    print(tag + ' set final length: ' + str(len(data_list)))




    # neg_sample = []
    #     while len(neg_sample) < self.neg:
    #         neg_sample_index = array[np.random.randint(10**8)]
    #         if neg_sample_index == word_index:
    #             continue
    #         neg_sample.append(neg_sample_index)
    #     return neg_sample

def SepKKBoxData(data_df, train_rate):
    print("Splitting data...")
    train = []
    train_user = []
    train_list = []
    valid = []
    valid_user = []
    valid_list = []
    test = []
    test_user = []
    test_list = []
    total = []
    total_list = []
    item_list = []
    random.seed(3)
    for idx, row in data_df.iterrows():
        user = row['user_id']
        item = row['song_id']
        user_count = row['user_counts']
        if item not in item_list:
            item_list.append(item)
        total.append([user, item])
        if user in train_user:
            train.append([user, item])
            # train_list.append([user, item, 1])
        elif user in valid_user:
            valid.append([user, item])
            # valid_list.append([user, item, 1])
        elif user in test_user:
            test.append([user, item])
            # test_list.append([user, item, 1])
        else:
            total_list.append(user)
            if 30 <= user_count <= 50:
                test_user.append(user)
                test.append([user, item])
                continue
            num = random.random()
            if num < train_rate:
                train_user.append(user)
                # train_list.append([user, item, 1])
                train.append([user, item])
            elif train_rate <= num:
                valid_user.append(user)
                # valid_list.append([user, item, 1])
                valid.append([user, item])
            # else:
            #     test_user.append(user)
            #     # test_list.append([user, item, 1])
            #     test.append([user, item])

    print('length of item list is: ' + str(len(item_list)))

    print('train ' + str(len(train)))
    print('train user ' + str(len(train_user)))
    train_user_df = pd.DataFrame(train_user, columns=['user_id'])
    train_user_df.to_csv('ProcessedData/train_user_df.csv', index=False)
    print('train_user_df.csv generated.')
    print('=======================================')
    print('valid ' + str(len(valid)))
    print('valid user ' + str(len(valid_user)))
    valid_user_df = pd.DataFrame(valid_user, columns=['user_id'])
    valid_user_df.to_csv('ProcessedData/valid_user_df.csv', index=False)
    print('valid_user_df.csv generated.')
    print('=======================================')
    print('test ' + str(len(test)))
    print('test user ' + str(len(test_user)))
    test_user_df = pd.DataFrame(test_user, columns=['user_id'])
    test_user_df.to_csv('ProcessedData/test_user_df.csv', index=False)
    print('test_user_df.csv generated.')
    print('=======================================')
    print('total ' + str(len(total)))
    print('total user ' + str(len(total_list)))
    print('Data is split.')
    print('=======================================')
    return PreProcessData(train, item_list, 4, 'train'), PreProcessData(valid, item_list, 4, 'valid'), PreProcessData(test, item_list, 20, 'test')

def OneHotEncoding(data_df):
    sparse_feature = data_df[['city', 'gender', 'genre_ids', 'artist_name', 'composer', 'lyricist', 'language']]

    le = LabelEncoder()
    for col in sparse_feature.columns:
        print(col)
        sparse_feature[col] = le.fit_transform(sparse_feature[col].values)
    sparse_feature.to_csv('sparse_feature_label_encoder.txt', index = False, header = False, sep = ',')

    sparse_feature_label = pd.read_table('../Logisitic Regression/sparse_feature_label_encoder.txt', encoding='utf-8', delimiter=',', header=None)
    sparse_feature_label = sparse_feature_label.values
    a_dis = sparse_feature_label.tolist()
    one = OneHotEncoder()
    one.fit(a_dis)
    b_dis = one.transform(a_dis).toarray()
    np.savetxt('sparse_feature_onehot_encoder.txt', b_dis, encoding='utf-8', delimiter=',', fmt='%d')
