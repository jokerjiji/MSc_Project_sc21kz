import pandas as pd

from Itemcf.Data_Preprocessing_test import OneHotEncoding

train_filtered_df = pd.read_csv('../ProcessedData/merged_train_df_final.csv', low_memory=False)
OneHotEncoding(train_filtered_df)