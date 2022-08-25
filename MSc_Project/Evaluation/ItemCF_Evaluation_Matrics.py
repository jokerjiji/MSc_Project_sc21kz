from pandas._libs.internals import defaultdict

def Hit_In_TopK(test_set, ItemCF):
    record_list = []
    hit_1_dict = defaultdict(int)
    hit_5_dict = defaultdict(int)
    hit_10_dict = defaultdict(int)
    hit_20_dict = defaultdict(int)
    for user_id in test_set:
        if user_id in record_list:
            continue
        else:
            record_list.append(user_id)
            recommend_dict = ItemCF.Recommend(user_id, 20, 80)
            recommend_list = list(recommend_dict.keys())
            unknown_history = test_set[user_id][10:]  # list
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



def ItemCF_Test(test_set, ItemCF):
    hit_k_dict = defaultdict(int)
    hit_k_avg = 0
    for user, items in test_set.items():
        items_count = len(items)
        recommend_dict = ItemCF.Recommend(user, 5, 80)
        # print(recommend_dict)
        # rec_count = len(recommend_dict)
        rec_right_count = 0
        if recommend_dict:
            for rec_item in recommend_dict:
                if rec_item in items:
                    rec_right_count = rec_right_count +1
        if items_count > 1:
            hit_k = rec_right_count / items_count
            hit_k_dict[user] = hit_k
            hit_k_avg += hit_k
    hit_k_avg = hit_k_avg / len(hit_k_dict)
    print(hit_k_avg)
    return hit_k_dict

def ItemCF_Sample_Test(user_id, train_set, valid_set, test_set, ItemCF, train_filtered_df):
    print('User have listened:')
    if user_id in test_set:
        SongName(test_set[user_id][:10], train_filtered_df)
    print('=======================================')


    recommend_dict = ItemCF.Recommend(user_id, 10, 80)
    # print(recommend_dict)
    print('Recommendation results: ')
    SongName(recommend_dict, train_filtered_df)
    print('=======================================')

def SongName(song_id_dict, train_filtered_df):
    # print(song_id_dict)
    for index, item in enumerate(song_id_dict):
        song = train_filtered_df.loc[train_filtered_df['song_id'] == item]
        print(str(index) + ': ' + song.iloc[0]['name'])

def Precision_Recall_Fscore(test_set, ItemCF):
    print('Evaluation running...')
    record_list = []
    precision_dict = defaultdict(float)
    recall_dict = defaultdict(float)
    f1score_dict = defaultdict(float)
    precision_avg = .0
    recall_avg = .0
    f1score_avg = .0
    for user_id in test_set:
        if user_id in record_list:
            continue
        else:
            record_list.append(user_id)
            recommend_dict = ItemCF.Recommend(user_id, 20, 80)
            recommend_list = recommend_dict.keys()
            recommend_list_length = len(recommend_list)
            unknown_history = test_set[user_id][10:]  # list
            unknown_history_len = len(unknown_history)
            if unknown_history_len < 1:
                print(user_id + 'unknown_history_len < 1')
                continue

            # for N in range(1, 5, 10, 20):  # 之后完善
            #     recommend_list_temp = recommend_list[:N]

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
                f1score = (2 * precision * recall)/(precision + recall)

            if precision == 0:
                print(user_id)

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

    return precision_avg, recall_avg, f1score_avg







