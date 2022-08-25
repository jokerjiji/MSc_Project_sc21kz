import pandas as pd
from pandas._libs.internals import defaultdict
from operator import itemgetter
import json

def overall_evaluation(result_df):
    result_dict = defaultdict(float)
    TP_overall = 0
    TN_overall = 0
    FP_overall = 0
    FN_overall = 0

    for i, row in result_df.iterrows():
        if row['label'] == 1:
            if row['predict'] == row['label']:
                TP_overall += 1
                continue
            elif row['predict'] != row['label']:
                FN_overall += 1
                continue
        elif row['label'] == 0:
            if row['predict'] == row['label']:
                TN_overall += 1
                continue
            elif row['predict'] != row['label']:
                FP_overall += 1
                continue

    precision_overall = TP_overall/(TP_overall + FP_overall)
    recall_overall = TP_overall/(TP_overall + FN_overall)
    if ((precision_overall == 0) & (recall_overall == 0)):
        f1score_overall = .0
    else:
        f1score_overall = (2 * precision_overall * recall_overall) / (precision_overall + recall_overall)

    print('precision_overall: ' + str(precision_overall))
    print('recall_overall: ' + str(recall_overall))
    print('f1score_overall: ' + str(f1score_overall))

def user_granularity(result_df, N):
    user_recommend = {}
    for i, row in result_df.iterrows():
        user_recommend.setdefault(row['user_id'], {})
        # if row['output'] >= 0.5:
        #     user_recommend[row['user_id']][row['song_id']] = row['output']
        user_recommend[row['user_id']][row['song_id']] = row['output']

    dict0 = dict()
    for user_id, user_dict in user_recommend.items():
        dict0[user_id] = dict(sorted(user_dict.items(), key=itemgetter(1), reverse=True)[:N])
    return dict0

def Hit_In_TopK(result_df):
    hit_1_dict = defaultdict(int)
    hit_5_dict = defaultdict(int)
    hit_10_dict = defaultdict(int)
    hit_20_dict = defaultdict(int)
    record_list = []
    user_recommends_dict = user_granularity(result_df, 20)
    f3 = open('../ProcessedData/ItemCFData/test_set.json', 'r')
    test_set = json.load(f3)
    # for user_id in test_set:
    #     if user_id in record_list:
    #         continue
    #     else:
    #         record_list.append(user_id)

    for user_id, user_dict in user_recommends_dict.items():
        if user_id in record_list:
            continue
        else:
            record_list.append(user_id)
            recommend_list = list(user_dict.keys())
            unknown_history = test_set[user_id][10:]
            unknown_history_len = len(unknown_history)

            top_1 = recommend_list[:1]
            top_5 = recommend_list[:5]
            top_10 = recommend_list[:10]
            top_20 = recommend_list[:20]
            if len(list(set(unknown_history) & set(top_1))) != 0:
                hit_1_dict[user_id] = 1
            if len(list(set(unknown_history) & set(top_5))) != 0:
                hit_5_dict[user_id] = 1
            if len(list(set(unknown_history) & set(top_10))) != 0:
                hit_10_dict[user_id] = 1
            if len(list(set(unknown_history) & set(top_20))) != 0:
                hit_20_dict[user_id] = 1
    hit_rate_1 = len(hit_1_dict) / len(record_list)
    hit_rate_5 = len(hit_5_dict) / len(record_list)
    hit_rate_10 = len(hit_10_dict) / len(record_list)
    hit_rate_20 = len(hit_20_dict) / len(record_list)
    print('Hit ratio in top1: ' + str(hit_rate_1))
    print('Hit ratio in top5: ' + str(hit_rate_5))
    print('Hit ratio in top10: ' + str(hit_rate_10))
    print('Hit ratio in top20: ' + str(hit_rate_20))
    return hit_rate_1, hit_rate_5, hit_rate_10, hit_rate_20

def TopN_Precision_Recall_F1score(result_df, N):
    user_recommends_dict = user_granularity(result_df, N)
    f3 = open('../ProcessedData/ItemCFData/test_set.json', 'r')
    test_set = json.load(f3)
    precision_dict = defaultdict(float)
    recall_dict = defaultdict(float)
    f1score_dict = defaultdict(float)
    precision_avg = .0
    recall_avg = .0
    f1score_avg = .0
    for user_id, user_dict in user_recommends_dict.items():
        recommend_list = user_dict.keys()
        unknown_history = test_set[user_id][10:]
        unknown_history_len = len(unknown_history)
        if unknown_history_len < 1:
            print(user_id + 'unknown_history_len < 1')
            continue
        TP = len(list(set(unknown_history) & set(recommend_list)))
        # print('TP ' + str(TP))
        # TN = list(set(unknown_history) & set(recommend_list))
        FP = len(list(set(recommend_list) - set(unknown_history)))
        # print('FP ' + str(FP))
        FN = len(list(set(unknown_history) - set(recommend_list)))
        # print('FN ' + str(FN))

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        if ((precision == 0) & (recall == 0)):
            f1score = .0
        else:
            f1score = (2 * precision * recall) / (precision + recall)

        if precision == 0:
            print(user_id)
            print(recommend_list)
            print(unknown_history)
            print(unknown_history_len)
            print("========================================")

        precision_avg += precision
        recall_avg += recall
        f1score_avg += f1score

        precision_dict[user_id] = precision
        recall_dict[user_id] = recall
        f1score_dict[user_id] = f1score

    precision_avg = precision_avg / len(precision_dict)
    recall_avg = recall_avg / len(recall_dict)
    f1score_avg = f1score_avg / len(f1score_dict)

    print('Average precision: ' + str(precision_avg))
    print('Average recall: ' + str(recall_avg))
    print('Average f1-score: ' + str(f1score_avg))
    print('Evaluation completed.')
    print('=======================================')

if __name__ == "__main__":
    result_df = pd.read_csv('../WideDeep/Results/wild&deep_result_df5.csv', low_memory=False)
    # Hit_In_TopK(result_df)
    # overall_evaluation(result_df)
    # TopN_Precision_Recall_F1score(result_df, 1)
    # TopN_Precision_Recall_F1score(result_df, 5)
    TopN_Precision_Recall_F1score(result_df, 10)
    # TopN_Precision_Recall_F1score(result_df, 20)


