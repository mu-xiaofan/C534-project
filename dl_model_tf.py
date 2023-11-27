import os
import math
import tqdm
import torch
import argparse
import data_preprocess
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
from utils import *


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TransformerModel(nn.Module):
    def __init__(self, n_feature, len_sw, n_class, d_model, nhead, num_layers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder_layers = TransformerEncoderLayer(d_model, nhead, d_model, dropout)
        self.transformer_encoder = TransformerEncoder(self.encoder_layers, num_layers)
        self.encoder = nn.Linear(n_feature * len_sw, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, n_class)

    def forward(self, x):
        x = self.encoder(x.view(x.size(0), -1)) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, self.src_mask)
        x = self.decoder(x)
        return F.log_softmax(x, dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.5, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

def evaluate(model, data_loader):
    model.eval()
    with torch.no_grad():
        predicted_labels = []
        true_labels = []
        lengths = []

        # test loop
        for sample, target in data_loader:
            sample, target = sample.to(DEVICE), target.to(DEVICE)
            output = model(sample)
            _, predicted = torch.max(output.data, 1)

            # collect predict and real labels
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(target.cpu().numpy())
            lengths.extend([sample.shape[1]] * sample.shape[0])  # assume length is the duration of time

        # transform list to Tensor
        predicted_labels = torch.tensor(predicted_labels)
        true_labels = torch.tensor(true_labels)
        lengths = torch.tensor(lengths)

        # calculate measure metrics
        event_acc, event_miF, event_maF, frame_acc, frame_miF, frame_maF = measure_event_frame(predicted_labels, lengths, true_labels)

    return event_acc, event_miF, event_maF, frame_acc, frame_miF, frame_maF


def train_transformer(model, optimizer, train_loader, val_loader, n_epoch, result_name):
    acc_all = []
    results = []
    best_metrics = {'e_acc': 0, 'e_miF': 0, 'e_maF': 0, 'f_acc': 0, 'f_miF': 0, 'f_maF': 0, 'iter': 0}
    """
    best_val_acc = 0
    early_stop_count = 0
    EARLY_STOP_THRESHOLD = 30
    """


    for e in range(n_epoch):
        model.train()
        total_loss, correct, total = 0, 0, 0

        # train
        for index, (sample, target) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            sample, target = sample.to(DEVICE), target.to(DEVICE)
            output = model(sample)
            loss = nn.CrossEntropyLoss()(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

        acc_train = 100. * correct / total
        tqdm.tqdm.write(f'Epoch: [{e+1}/{n_epoch}], Training loss: {total_loss / len(train_loader):.4f}, Training Accuracy: {acc_train:.2f}%')

        # Validation loop
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for val_sample, val_target in val_loader:
                val_sample, val_target = val_sample.to(DEVICE), val_target.to(DEVICE)
                val_output = model(val_sample)
                _, val_predicted = torch.max(val_output.data, 1)
                val_total += val_target.size(0)
                val_correct += (val_predicted == val_target).sum().item()
        val_acc = 100. * val_correct / val_total
        tqdm.tqdm.write(f'Epoch: [{e+1}/{n_epoch}], Validation Accuracy: {val_acc:.2f}%')

        """
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_count = 0
            # save model
            torch.save(model.state_dict(), f'{result_name}_best_model.pth')
        else:
            early_stop_count += 1

        # TODO: add to settings
        if early_stop_count >= EARLY_STOP_THRESHOLD:
            print("Early stopping triggered.")
            break
        """

        # test process
        event_acc, event_miF, event_maF, frame_acc, frame_miF, frame_maF = evaluate(model, val_loader)
        acc_all.append([event_acc, event_miF, event_maF, frame_acc, frame_miF, frame_maF])
        results.append([acc_train, event_acc, event_miF, event_maF, frame_acc, frame_miF, frame_maF])

        # update best params
        if event_miF > best_metrics['e_miF']:
            best_metrics.update({'e_acc': event_acc, 'e_miF': event_miF, 'e_maF': event_maF, 'f_acc': frame_acc, 'f_miF': frame_miF, 'f_maF': frame_maF, 'iter': e + 1})

    # save result
    np.savetxt(result_name, np.array(results), fmt='%.2f', delimiter=',')
    print(result_name)
    print(best_metrics)
    if not os.path.exists('results/' + args.model_name):
        os.makedirs('results/' + args.model_name)
    if not os.path.isfile(result_name):
        with open(result_name, 'w') as my_empty_csv:
            pass

    with open('results/{}/{}_best_metrics.txt'.format(args.model_name, args.model_name), 'a') as f:
        f.write('\n'.join([f'{k}: {v}' for k, v in best_metrics.items()]))
    return best_metrics


parser = argparse.ArgumentParser(description='Transformer model training')

parser.add_argument('--n_feature', type=int, default=9, help='Number of features')
parser.add_argument('--len_sw', type=int, default=32, help='Length of sliding window')
parser.add_argument('--n_class', type=int, default=2, help='Number of classes')
parser.add_argument('--d_model', type=int, default=512, help='Dimension of Transformer model')
parser.add_argument('--nhead', type=int, default=8, help='Number of heads in multihead attention')
parser.add_argument('--num_layers', type=int, default=3, help='Number of Transformer layers')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')

parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
parser.add_argument('--n_epoch', type=int, default=100, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--dataset', type=str, default='dg', help='Dataset name')
parser.add_argument('--model_name', type=str, default='dg', help='transformer')

args = parser.parse_args()
train_loader, val_loader, test_loader = data_preprocess.load_dataset_dl(batch_size=args.batch_size, SLIDING_WINDOW_LEN=32, SLIDING_WINDOW_STEP=16)

lr_options = [1e-4, 1e-5]
dropout_options = [0.5, 0.4, 0.3]
best_val_acc = 0
best_model_info = {}

for lr in lr_options:
    for dropout in dropout_options:
        print(f"Training with lr={lr}, dropout={dropout}")
        model = TransformerModel(args.n_feature, args.len_sw, args.n_class, args.d_model, args.nhead, args.num_layers, dropout).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        result_name = f'transformer_lr{lr}_dropout{dropout}'

        current_best_metrics = train_transformer(model, optimizer, train_loader, test_loader, args.n_epoch, result_name)

        if current_best_metrics['e_miF'] > best_val_acc:
            best_val_acc = current_best_metrics['e_miF']
            best_model_info = {
                'model_state': model.state_dict(),
                'lr': lr,
                'dropout': dropout,
                'result_name': result_name
            }

# 保存最佳模型
if best_model_info:
    torch.save(best_model_info['model_state'], f'{best_model_info["result_name"]}_best_model.pth')

# 测试集评估
model.load_state_dict(torch.load(f'{best_model_info["result_name"]}_best_model.pth'))
test_res = evaluate(model, val_loader)
print("Test result:", test_res)

"""
model = TransformerModel(args.n_feature, args.len_sw, args.n_class, args.d_model, args.nhead, args.num_layers, args.dropout)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
train_loader, val_loader, test_loader = data_preprocess.load_dataset_dl(batch_size=args.batch_size, SLIDING_WINDOW_LEN=32, SLIDING_WINDOW_STEP=16)

n_epoch = 100
n_batch = len(train_loader.dataset) // args.batch_size
result_name = 'transformer'

best_acc = train_transformer(model, optimizer, train_loader, test_loader, n_epoch, result_name)
"""
