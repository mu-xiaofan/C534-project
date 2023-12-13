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
import itertools
from utils import *


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

LOSS_FN_WEIGHT = 1e-5
# initialize
result = []
acc_all = []

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

def train_dg_fixed(model, optimizer, train_loader, test_loader, now_model_name, args):
    """
    Train function.
    """
    feature_dim = args.n_feature # Get feature dimension
    n_batch = len(train_loader.dataset) // args.batch_size # Calculate the batch size for every epoch
    criterion = nn.CrossEntropyLoss()
    criterion_ae = nn.MSELoss()

    # Train process
    for e in range(args.n_epoch):
        if e > 0 and e % 50 == 0:
            plot(result_name)

        model.train()
        correct, total_loss = 0, 0
        total = 0

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


            # Print training process
            if index % 20 == 0:
                tqdm.tqdm.write('Epoch: [{}/{}], Batch: [{}/{}], loss_total:{:.4f}'.format(
                    e + 1, args.n_epoch, index + 1, n_batch, loss.item()))

        # calculate accuracy
        acc_train = float(correct) * 100.0 / (args.batch_size * n_batch)
        tqdm.tqdm.write(
            'Epoch: [{}/{}], loss: {:.4f}, train acc: {:.2f}%'.format(e + 1, args.n_epoch, total_loss * 1.0 / n_batch, acc_train))

        # Testing
        model.train(False)
        with torch.no_grad():
            correct, total = 0, 0
            event_acc, event_miF, event_maF, frame_acc, frame_miF, frame_maF = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            predicted_label_segment, lengths_varying_segment, true_label_segment = torch.LongTensor(), torch.LongTensor(), torch.LongTensor()
            for sample, target in test_loader:
                sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
                now_len = sample.shape[1]
                # this line would cause error since the batch of last iteration does not have batch_size entries. so use DropLast = True when prep for dataloader
                sample = sample.view(-1, feature_dim, now_len)
                sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()

                output = model(sample)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum()
                lengths_varying = [sample.shape[2]] * sample.shape[0]
                lengths_varying = torch.LongTensor(lengths_varying)
                predicted_label_segment = torch.LongTensor(torch.cat((predicted_label_segment, predicted.cpu()), dim=0))
                lengths_varying_segment = torch.LongTensor(torch.cat((lengths_varying_segment, lengths_varying), dim=0))
                true_label_segment = torch.LongTensor(torch.cat((true_label_segment, target.cpu()), dim=0))


        # calculate different measurements
        # event: accuracy, micro-F1, macro-F1
        # frame: accuracy, micro-F1, macro-F1
        event_acc, event_miF, event_maF, frame_acc, frame_miF, frame_maF = measure_event_frame(predicted_label_segment, lengths_varying_segment, true_label_segment)
        acc_all.append([event_acc, event_miF, event_maF, frame_acc, frame_miF, frame_maF])
        acc_all_T = np.array(acc_all).T.tolist()

        best_e_miF = max([row[1] for row in acc_all])
        best_iter = acc_all_T[1].index(best_e_miF) + 1

        best_e_acc = acc_all[best_iter-1][0]
        best_e_maF = acc_all[best_iter-1][2]
        best_f_acc = acc_all[best_iter-1][3]
        best_f_miF = acc_all[best_iter-1][4]
        best_f_maF = acc_all[best_iter-1][5]
        if sum(predicted_label_segment) == 0:
            print('Note: All predicted labels are 0 in this epoch!\n')

        tqdm.tqdm.write(
            'Epoch: [{}/{}], e acc:{:.2f}%, e_miF:{:.2f}%, e maF:{:.2f}%, f acc:{:.2f}%, f miF:{:.2f}%, f maF:{:.2f}%, best acc:{:.2f}%, iter:{}'.format(
                e + 1, args.n_epoch, event_acc, event_miF, event_maF, frame_acc, frame_miF, frame_maF, best_e_acc,
                best_iter))

        # save result
        result.append([acc_train, event_acc, event_miF, event_maF, frame_acc, frame_miF, frame_maF, best_e_acc, best_iter])
        result_np = np.array(result, dtype=float)
        np.savetxt(result_name, result_np, fmt='%.2f', delimiter=',')

    return best_e_acc, best_e_miF, best_e_maF, best_f_acc, best_f_miF, best_f_maF, best_iter



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
parser.add_argument('--model_name', type=str, default='transformer', help='Model name')

param_space = {
    'n_feature': [9, 10],
    'len_sw': [32, 64],
    'n_class': [2, 3],
    'd_model': [512, 1024],
    'nhead': [8, 16],
    'num_layers': [3, 6],
    'dropout': [0.5, 0.7],
    'batch_size': [64, 128],
    'n_epoch': [100, 150],
}

def train_and_test(args):
    train_loader, val_loader, test_loader = data_preprocess.load_dataset_dl(batch_size=args.batch_size, SLIDING_WINDOW_LEN=args.len_sw, SLIDING_WINDOW_STEP=16)
    model = TransformerModel(args.n_feature, args.len_sw, args.n_class, args.d_model, args.nhead, args.num_layers, args.dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    result_name = 'results/' + args.model_name + '/' + str(args.n_epoch) + '_' + str(args.batch_size) + '_' + args.model_name + '.csv'
    train_dg_fixed(model, optimizer, train_loader, val_loader, result_name, args)

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for sample, target in test_loader:
            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
            output, _ = model(sample)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_accuracy

best_performance = 0
best_params = None
best_test_metrics = None
for combination in itertools.product(*param_space.values()):
    args = argparse.Namespace(**dict(zip(param_space.keys(), combination)))
    train_metrics, test_metrics = train_and_test(args)
    performance = test_metrics[0]

    if performance > best_performance:
        best_performance = performance
        best_params = args
        best_test_metrics = test_metrics


print("Best performance: ", best_performance)
print("Best parameter: ", vars(best_params))