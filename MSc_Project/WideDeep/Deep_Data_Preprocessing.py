import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def DataMerging():
    song_file_path = "C:/Users/35402/OneDrive - University of Leeds/Msc Project/Datasets/kkbox-music-recommendation-challenge/songs.csv"
    song_detail_file_path = "C:/Users/35402/OneDrive - University of Leeds/Msc Project/Datasets/kkbox-music-recommendation-challenge/song_extra_info.csv"
    user_file_path = "C:/Users/35402/OneDrive - University of Leeds/Msc Project/Datasets/kkbox-music-recommendation-challenge/members.csv"

    # song
    song_df = pd.read_csv(song_file_path)
    song_df_notnull = song_df.dropna()

    # song detail
    song_detail_df = pd.read_csv(song_detail_file_path)
    song_detail_df = song_detail_df[['song_id', 'name']]
    merged_song_df = song_df_notnull.merge(song_detail_df, on=['song_id'])

    #user
    user_df = pd.read_csv(user_file_path)
    user_notnull = user_df.dropna()
    user_cleaned = user_notnull[(user_notnull['bd'] != 0)].reset_index(drop=True)
    user_cleaned.rename(columns={'msno': 'user_id'}, inplace=True)

    train_df = pd.read_csv('../ProcessedData/DeepData/train_df.csv', low_memory=False)
    valid_df = pd.read_csv('../ProcessedData/DeepData/valid_df.csv', low_memory=False)
    test_df = pd.read_csv('../ProcessedData/DeepData/test_df.csv', low_memory=False)
    data_df = pd.concat((train_df, valid_df, test_df))

    data_df = data_df.merge(user_cleaned, on=['user_id'])
    data_df = data_df.merge(merged_song_df, on=['song_id'])

    # print(len(data_df))
    # data_df_notnull = data_df.dropna()
    # print(len(data_df_notnull))
    # print(data_df_notnull.columns)

    # 分开测试集和训练集
    train = pd.read_csv('../ProcessedData/DeepData/train_user_df.csv', low_memory=False)
    valid = pd.read_csv('../ProcessedData/DeepData/valid_user_df.csv', low_memory=False)
    test = pd.read_csv('../ProcessedData/DeepData/test_user_df.csv', low_memory=False)

    train_set = train.merge(data_df, on=['user_id'])
    valid_set = valid.merge(data_df, on=['user_id'])
    test_set = test.merge(data_df, on=['user_id'])

    # 保存文件
    train_set.reset_index(drop=True, inplace=True)
    valid_set.reset_index(drop=True, inplace=True)
    test_set.reset_index(drop=True, inplace=True)

    # train = data_df[:train_df.shape[0]]
    # valid = data_df[train_df.shape[0]:(train_df.shape[0] + valid_df.shape[0])]
    # test = data_df[(train_df.shape[0] + valid_df.shape[0]):]

    train_set.to_csv('../ProcessedData/DeepData/train_set_raw.csv', index=0)
    valid_set.to_csv('../ProcessedData/DeepData/val_set_raw.csv', index=0)
    test_set.to_csv('../ProcessedData/DeepData/test_set_raw.csv', index=0)

    print(len(train_set))
    print(len(valid_set))
    print(len(test_set))

def DataPreprocess():
    # import data

    train_df = pd.read_csv('../ProcessedData/DeepData/train_set_raw.csv', low_memory=False)
    valid_df = pd.read_csv('../ProcessedData/DeepData/val_set_raw.csv', low_memory=False)
    test_df = pd.read_csv('../ProcessedData/DeepData/test_set_raw.csv', low_memory=False)
    data_df = pd.concat((train_df, valid_df, test_df))


    print(data_df.shape)
    data_df = data_df.reset_index(drop=True).rename(columns={'target': 'label'})

    del data_df['registered_via']
    del data_df['registration_init_time']
    del data_df['expiration_date']
    del data_df['name']
    # del data_df['user_counts']
    # del data_df['song_counts']
    del data_df['user_id']
    del data_df['song_id']

    print(data_df.columns)

    # Separate feature categories
    sparse_feas = ['city', 'gender', 'genre_ids', 'artist_name', 'composer', 'lyricist', 'language']
    dense_feas = ['bd', 'song_length']

    # Fill missing values
    data_df[sparse_feas] = data_df[sparse_feas].fillna('-1')
    data_df[dense_feas] = data_df[dense_feas].fillna(0)

    # Category feature encoding
    for feat in sparse_feas:
        le = LabelEncoder()
        data_df[feat] = le.fit_transform(data_df[feat])

    # MinMax Normalization
    mms = MinMaxScaler()
    data_df[dense_feas] = mms.fit_transform(data_df[dense_feas])

    train_set = data_df[:train_df.shape[0]]
    valid_set = data_df[train_df.shape[0]:(train_df.shape[0] + valid_df.shape[0])]
    test_set = data_df[(train_df.shape[0] + valid_df.shape[0]):]

    # del test_set['label']


    # print(train_set['Label'].value_counts())
    # print(valid_set['Label'].value_counts())

    # 保存文件
    train_set.reset_index(drop=True, inplace=True)
    valid_set.reset_index(drop=True, inplace=True)
    test_set.reset_index(drop=True, inplace=True)

    print(len(train_set))
    print(len(valid_set))
    print(len(test_set))

    train_set.to_csv('../ProcessedData/DeepData/train_set.csv', index=0)
    valid_set.to_csv('../ProcessedData/DeepData/val_set.csv', index=0)
    test_set.to_csv('../ProcessedData/DeepData/test_set.csv', index=0)

def getTrainData(filename, feafile):
    df = pd.read_csv(filename)
    print(df.columns)

    # C开头的列代表稀疏特征，I开头的列代表的是稠密特征
    dense_features_col = ['bd', 'song_length']

    # 这个文件里面存储了稀疏特征的最大范围，用于设置Embedding的输入维度
    fea_col = np.load(feafile, allow_pickle=True)
    sparse_features_col = []
    for f in fea_col[1]:
        sparse_features_col.append(f['feat_num'])

    data, labels = df.drop(columns='label').values, df['label'].values

    return data, labels, dense_features_col, sparse_features_col

def getTestData(filename):
    df = pd.read_csv(filename)
    print(df.columns)

    data, labels = df.drop(columns='label').values, df['label'].values

    return data, labels

if __name__ == "__main__":
    DataMerging()
    DataPreprocess()