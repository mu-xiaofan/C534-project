import os
import matplotlib
import data_preprocess
import torch
import torch.nn as nn
import tqdm
import argparse
import itertools
from utils import *
import torch.nn.functional as F


LOSS_FN_WEIGHT = 1e-5


class DDNN(nn.Module):
    def __init__(self, args):
        super(DDNN, self).__init__()
        self.n_lstm_hidden = args.n_lstm_hidden  # LSTM hidden layer number
        self.n_lstm_layer = args.n_lstm_layer    # LSTM layer number

        self.n_feature = args.n_feature  # feature number
        self.len_sw = args.len_sw        # sliding window length
        self.n_class = args.n_class      # class number
        self.d_AE = args.d_AE            # Autoencoder's demension

        # Define encoder
        # Convert input data into a low-dimensional representation
        self.encoder = nn.Sequential(
            nn.Linear(self.n_feature * self.len_sw, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, self.d_AE))

        # Define decoder
        # Restore encoded low-dimensional data to original high-dimensional data
        self.decoder = nn.Sequential(
            nn.Linear(self.d_AE, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, self.n_feature * self.len_sw),
            nn.Tanh())

        # RNN part
        # LSTM
        self.lstm = nn.LSTM(self.n_feature, self.n_lstm_hidden, self.n_lstm_layer, batch_first=True)
        self.lstm_spatial = nn.LSTM(self.len_sw, self.n_lstm_hidden, self.n_lstm_layer, batch_first=True)

        # Convolutional layer
        # A series of convolutional and pooling layers are used to extract features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.n_feature, out_channels=1024, kernel_size=(1, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2))

        # Full connection layer
        # Conbine features and output the final results
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=(2*self.n_lstm_hidden + self.d_AE + 64), out_features=1000),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=1000, out_features=500),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=500, out_features=self.n_class))

    def forward(self, x):
        out_encoder = self.encoder(x.view(x.size(0), -1))
        out_decoder = self.decoder(out_encoder)

        # Process time-series data using LSTM
        out_rnn, _ = self.lstm(x.view(x.shape[0], -1, self.n_feature)) # (batch_size, sequence_length, n_feature)
        out_rnn = out_rnn[:, -1, :] # Last time step as output

        # Process space data using LSTM
        out_rnn_spatial, _ = self.lstm_spatial(x.view(x.shape[0], self.n_feature, -1))  # (batch_size, n_feature, sequence_length)
        out_rnn_spatial = out_rnn_spatial[:, -1, :] # Last time step as output

        # Convolutional layer: extract features gradually
        out_conv1 = self.conv1(x.view(-1, x.shape[1], 1, x.shape[2]))
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv4 = out_conv4.reshape(-1, out_conv4.shape[1] * out_conv4.shape[3]) # reshape to fit full connection layer

        # combine all the features
        out_combined = torch.cat((out_encoder, out_rnn, out_rnn_spatial, out_conv4), dim=1)  # (batch_size, combined_features)
        # Get final output using full connection layer
        out_combined = self.fc1(out_combined)
        out_combined = self.fc2(out_combined)
        out_combined = self.fc3(out_combined)
        # Apply the Softmax function to get the final class probability distribution
        out_combined = F.softmax(out_combined, dim=1)
        return out_combined, out_decoder

# Use GPU if available
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
matplotlib.use('Agg')  # using agg as backend for drawing plots

# initialize
result = []
acc_all = []


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

        # iterate over training data
        for index, (sample, target) in enumerate(train_loader):
            # transfer data to device and convert to the proper format
            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
            now_len = sample.shape[1]
            sample = sample.view(-1, feature_dim, now_len)
            sample, target = sample.to(DEVICE).float(), target.to(DEVICE).long()
            output, out_decoder = model(sample)

            # ==== Calculate loss ====
            # Cross entropy loss function
            loss_classify = criterion(output, target)
            # Mean squared error loss function
            loss_ae = criterion_ae(sample.view(sample.size(0), -1), out_decoder)
            # Maximum mean discrepancy loss
            loss_mmd = mmd_custorm(sample.view(sample.size(0), -1), out_decoder, [args.sigma])
            loss_mmd = loss_mmd.to(DEVICE).float()
            # Calculated weighted loss
            loss = loss_classify + LOSS_FN_WEIGHT * loss_ae + args.weight_mmd * loss_mmd

            # ==== Back propagation ====
            # Set gradient as zero
            optimizer.zero_grad()
            # Calculate gradient
            loss.backward()
            # Update params based on gradient
            optimizer.step()

            # Update statistical data
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum()

            # Print training process
            if index % 20 == 0:
                tqdm.tqdm.write('Epoch: [{}/{}], Batch: [{}/{}], loss_ae:{:.4f}, loss_mmd:{:.4f}, loss_classify:{:.4f}, loss_total:{:.4f}'.format(
                    e + 1, args.n_epoch, index + 1, n_batch,
                    loss_ae.item(), loss_mmd.item(), loss_classify.item(), loss.item()))

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

                output, out_decoder = model(sample)
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


parser = argparse.ArgumentParser(description='argument setting of network')
parser.add_argument('--now_model_name', type=str, default='DDNN', help='the type of model, default DDNN')
parser.add_argument('--n_lstm_layer', type=int, default=1, help='number of lstm layers,default 2')
parser.add_argument('--n_lstm_hidden', type=int, default=64, help= 'number of lstm hidden dim, default 64')
parser.add_argument('--batch_size', type=int, default=64, help='batch size of training')
parser.add_argument('--n_epoch', type=int, default=100, help='number of training epochs')
parser.add_argument('--model_name', type=str, default='lstm', help='name of model')

parser.add_argument('--n_feature', type=int, default=9, help='name of feature dimension')
parser.add_argument('--len_sw', type=int, default=32, help='length of sliding window')
parser.add_argument('--n_class', type=int, default=2, help='number of class')
parser.add_argument('--d_AE', type=int, default=50, help='dim of AE')
parser.add_argument('--sigma', type=float, default=1, help='parameter of mmd')
parser.add_argument('--weight_mmd', type=float, default=1.0, help='weight of mmd loss')


torch.manual_seed(10)
args = parser.parse_args()


param_space = {
    'n_lstm_hidden': [32, 64, 128],
    'n_lstm_layer': [1, 2, 3],
    'batch_size': [32, 64, 128],
}

def train_and_test(args):
    train_loader, val_loader, test_loader = data_preprocess.load_dataset_dl(batch_size=args.batch_size, SLIDING_WINDOW_LEN=32, SLIDING_WINDOW_STEP=16)
    model = DDNN(args).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    result_name = 'results/' + args.model_name + '/' + str(args.n_epoch) + '_' + str(args.batch_size) + '_' + args.now_model_name + '_' + str(args.n_lstm_hidden) + '_' + str(args.n_lstm_layer) + '.csv'
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
for combination in itertools.product(*param_space.values()):
    args = argparse.Namespace(**dict(zip(param_space.keys(), combination)))
    performance = train_and_test(args)
    if performance > best_performance:
        best_performance = performance
        best_params = args

print("Best performance: ", best_performance)
print("Best parameter: ", vars(best_params))

