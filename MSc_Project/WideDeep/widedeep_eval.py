import torch
import pandas as pd
from WideDeep.trainer import Trainer
from WideDeep.network import WideDeep
from Deep_Data_Preprocessing import getTestData, getTrainData
import torch.utils.data as Data
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np

widedeep_config = \
{
    'deep_dropout': 0,
    'embed_dim': 8, # 用于控制稀疏特征经过Embedding层后的稠密特征大小
    'hidden_layers': [256,128,64],
    'num_epoch': 10,
    'batch_size': 128,
    'lr': 1e-3,
    'l2_regularization': 1e-4,
    'device_id': 0,
    'use_cuda': True,
    'train_file': '../ProcessedData/DeepData/train_set.csv',
    'fea_file': '../Utils/fea_col.npy',
    'validate_file': '../ProcessedData/DeepData/val_set.csv',
    'test_file': '../ProcessedData/DeepData/test_set.csv',
    'model_name': 'TrainedModels/WideDeep5.model'
}

if __name__ == "__main__":
    ####################################################################################
    # WideDeep
    ####################################################################################
    training_data, training_label, dense_features_col, sparse_features_col = getTrainData(widedeep_config['train_file'], widedeep_config['fea_file'])
    # train_dataset = Data.TensorDataset(torch.tensor(training_data).float(), torch.tensor(training_label).float())
    test_data, test_labels = getTestData(widedeep_config['test_file'])
    test_dataset = Data.TensorDataset(torch.tensor(test_data).float(), torch.tensor(test_labels).float())
    test_df = pd.read_csv('../ProcessedData/DeepData/test_df.csv', low_memory=False)
    wideDeep = WideDeep(widedeep_config, dense_features_cols=dense_features_col, sparse_features_cols=sparse_features_col)

    ####################################################################################
    # model eval
    ####################################################################################
    wideDeep.eval()
    if widedeep_config['use_cuda']:
        wideDeep.loadModel(map_location=lambda storage, loc: storage.cuda(widedeep_config['device_id']))
        wideDeep.cuda()
    else:
        wideDeep.loadModel(map_location=torch.device('cpu'))

    # with torch.no_grad():
    #     data_loader = DataLoader(dataset=test_dataset, batch_size=widedeep_config['batch_size'], shuffle=False)
    #
    #     for x, labels in data_loader:
    #         x = Variable(x)
    #         labels = Variable(labels)
    #         if widedeep_config['use_cuda'] is True:
    #             x, labels = x.cuda(), labels.cuda()
    #         # loss, predicted = wideDeep(x)
    #         outputs = wideDeep(x)
    #         # print(outputs)

    y_pred_probs = wideDeep(torch.tensor(test_data).float().cuda())
    y_pred = torch.where(y_pred_probs > 0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))
    y_pred_probs = y_pred_probs.cpu().detach().numpy().flatten()
    y_pred = y_pred.cpu().int().numpy().flatten()
    df = pd.DataFrame({'user_id': test_df['user_id'], 'song_id': test_df['song_id'], 'output': y_pred_probs, 'predict': y_pred, 'label': test_labels})
    # df['column_name'] = pd.Series(arr)
    df.to_csv('Results/wild&deep_result_df5.csv', index=False)
    # print(type(y_pred))
    # y_pred = torch.where(y_pred_probs>0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))
    # print("Test Data CTR Predict...\n ", y_pred.view(-1))

