import base64
import os
from PIL import Image
from io import BytesIO
from datetime import datetime, timedelta
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from torch.utils.data import Dataset
from pytorch_forecasting.metrics import MAPE
from soft_dtw_cuda import SoftDTW

import settings
from data_source.db_query import get_cpu_query
from data_source.dataframe import hosts_timeseries
from data_source.retrieve_data import get_df
from output_images import generate_predictions_html, get_loss_img, compute_test_error, VERSION_NAME, GLOBAL_DIREC


def df_to_treated_tensor(df_wks):
    '''
    Equivalent to DataModule.setup() in lstm_modularized
    '''
    #interpolate values and get same date intervals for all
    dfs, hostnames =  hosts_timeseries(df_wks)
    dfs = [a.rename(h) for a,h in zip(dfs,hostnames)]
    dataset = pd.concat(dfs,axis=1)

    #scale values
    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataset_array_scaled = scaler.fit_transform(dataset)
    dataset_scaled = dataset.copy()
    dataset_scaled.loc[:,:] = dataset_array_scaled

    tensor_dataset = torch.tensor(dataset_scaled.values)

    return dataset, tensor_dataset, hostnames, scaler


def split_data(tensor):
    '''
        returns train, test dataframes.
        Obs:  (index where test starts is clearly len(train)+1)
    '''
    per = 0.7 #percentage train
    len_train = int(per*len(tensor))
    train = tensor[:len_train,:]
    len_test = int((len(tensor) - len_train)/2)
    test = tensor[len_train:len_train+len_test, :]
    valid = tensor[len_train+len_test: , :]
    return train, test, valid

class MyDataset(Dataset):
    def __init__(self, data, q):
        self.data = torch.Tensor(data)
        self.window = q

    def __len__(self):
        return len(self.data) -  self.window

    def __getitem__(self, index):
        x = self.data[index:index+self.window]
#         y = self.data[index+1: index + self.window + 1]
        y = self.data[index + self.window]
        return x , y
    
def get_batches_dataloader(data, window, batch_size):
    dataset = MyDataset(data.float(), q = window)
    return torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = False, drop_last=True)


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, drop_prob=0):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

#         self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, dropout=drop_prob, batch_first=True)

    def forward(self, inputs, hidden):
        #Embed input words
#         embedded = self.embedding(inputs)
        embedded = inputs
        #Pass the embedded word vectors into LSTM and return all outputs
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self, device,  batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device))

class BahdanauDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, drop_prob=0.1):
        super(BahdanauDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.drop_prob = drop_prob

        self.fc_hidden = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_encoder = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.weight = nn.Parameter(torch.FloatTensor(1, hidden_size))
        self.dropout = nn.Dropout(self.drop_prob)
        self.lstm = nn.LSTM(self.hidden_size+self.output_size, self.hidden_size, batch_first=True)
        self.classifier = nn.Linear(self.hidden_size, output_size)

    def forward(self, inputs, hidden, encoder_outputs):
        encoder_outputs = encoder_outputs.squeeze()
        #Embed input 
        embedded = inputs.view(1, -1)
        embedded = self.dropout(embedded)
        #Calculating Alignment Scores
        x = torch.tanh(self.fc_hidden(hidden[0])+self.fc_encoder(encoder_outputs))
        alignment_scores = x.bmm(self.weight.unsqueeze(2))  
        #Softmaxing alignment scores to get Attention weights
        attn_weights = F.softmax(alignment_scores.view(1,-1), dim=1)
       

        #Multiplying the Attention weights with encoder outputs to get the context vector
        context_vector = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        #Concatenating context vector with embedded input word
        output = torch.cat((embedded, context_vector[0]), 1).unsqueeze(0)
        #Passing the concatenated vector as input to the LSTM cell
        output, hidden = self.lstm(output, hidden)
        #Passing the LSTM output through a Linear layer acting as a classifier
        output = self.classifier(output[0])
        return output, hidden, attn_weights

class LuongDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, attention, n_layers=1, drop_prob=0.1):
        super(LuongDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.drop_prob = drop_prob

        #Our Attention Mechanism is defined in a separate class
        self.attention = attention

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.drop_prob)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size*2, self.output_size)
    
    def forward(self, inputs, hidden, encoder_outputs):
        #Embed input words
        embedded = self.embedding(inputs).view(1,1,-1)
        embedded = self.dropout(embedded)

        #Passing previous output word (embedded) and hidden state into LSTM cell
        lstm_out, hidden = self.lstm(embedded, hidden)

        #Calculating Alignment Scores - see Attention class for the forward pass function
        alignment_scores = self.attention(hidden[0], encoder_outputs)
        #Softmaxing alignment scores to obtain Attention weights
        attn_weights = F.softmax(alignment_scores.view(1,-1), dim=1)

        #Multiplying Attention weights with encoder outputs to get context vector
        context_vector = torch.bmm(attn_weights.unsqueeze(0),encoder_outputs)

        #Concatenating output from LSTM with context vector
        output = torch.cat((lstm_out, context_vector),-1)
        #Pass concatenated vector through Linear layer acting as a Classifier
        output = F.log_softmax(self.classifier(output[0]), dim=1)
        return output, hidden, attn_weights

class Attention(nn.Module):
    def __init__(self, hidden_size, method="dot"):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        #Defining the layers/weights required depending on alignment scoring method
        if method == "general":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)

        elif method == "concat":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
            self.weight = nn.Parameter(torch.FloatTensor(1, hidden_size))
  
    def forward(self, decoder_hidden, encoder_outputs):
        if self.method == "dot":
          #For the dot scoring method, no weights or linear layers are involved
          return encoder_outputs.bmm(decoder_hidden.view(1,-1,1)).squeeze(-1)
    
        elif self.method == "general":
            #For general scoring, decoder hidden state is passed through linear layers to introduce a weight matrix
            out = self.fc(decoder_hidden)
            return encoder_outputs.bmm(out.view(1,-1,1)).squeeze(-1)

        elif self.method == "concat":
            #For concat scoring, decoder hidden state and encoder outputs are concatenated first
            out = torch.tanh(self.fc(decoder_hidden+encoder_outputs))
            return out.bmm(self.weight.unsqueeze(-1)).squeeze(-1)


def train(encoder, 
        decoder, 
        encoder_optimizer, 
        decoder_optimizer, 
        train_all, 
        train_window, 
        device, 
        hostnames,
        epochs,
        criterion):

    teacher_forcing_prob = 0.5
    encoder.train()
    decoder.train()

    data = get_batches_dataloader(train_all, train_window, batch_size = 1)

    avg_loss = []
    for epoch in range(epochs):
        print("****EPOCH ", epoch)
        samples_errors = []
        for  x, y in data:
            loss = 0.
            h = encoder.init_hidden(device)
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            if (len(x.shape)==2):
                x = x.unsqueeze(0)
            inp = x.to(device)
            encoder_outputs, h = encoder(inp,h)

            #First decoder input will be the SOS token
            decoder_input = torch.zeros(1 , len(hostnames), 1, device=device)
            #First decoder hidden state will be last encoder hidden state
            decoder_hidden = h
            output = []
            teacher_forcing = True if random.random() < teacher_forcing_prob else False
            
            decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_outputs)

            top_value, top_index = decoder_output.topk(1)
            decoder_input = decoder_output
            output.append(top_index.item())
            #Calculate the loss of the prediction against the actual word
            loss = criterion(decoder_output.view(1,-1), y.to(device))
            loss.backward()
            samples_errors.append(loss.item())
            encoder_optimizer.step()
            decoder_optimizer.step()
        avg_loss.append(np.sum(samples_errors)/data.__len__())

    return avg_loss

def test(encoder, 
        decoder, 
        train_window, 
        train_all, 
        test_all,
        hostnames,
        device):


    encoder.eval()
    decoder.eval()

    prediction_range = 3

    test_inputs = (train_all[-train_window:])
    preds = torch.tensor([])
    all_preds = torch.tensor([])

    # if device == 'cuda':
    test_inputs = test_inputs.cuda().to(device)
    preds = preds.cuda().to(device)
    all_preds = all_preds.cuda().to(device)
    test_all = test_all.cuda().to(device)

    h = encoder.init_hidden(device)

    for i in range(len(test_all)):
        #get last predictions by getting values in preds tensor and the number of ground truth values needed to complete train_window
        #put first dim as batch_dim (input.shape== batch_dim,seq_len,num_feat)
        seq = torch.cat((test_inputs[-(train_window - preds.shape[0]):], preds),0).unsqueeze(0).float()
        seq = seq.cuda().to(device)

        with torch.no_grad():
            encoder_outputs, h = encoder(seq,h)
            decoder_input = torch.zeros(1 , len(hostnames), 1, device=device)
            decoder_hidden = h
            decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = decoder_output

            test_preds = decoder_output.view(1,-1).cuda().to(device)

            cuda_check = test_preds.is_cuda
            if cuda_check:
                get_cuda_device = test_preds.get_device()
                print("test_preds device: ", get_cuda_device)

            cuda_check = preds.is_cuda
            if cuda_check:
                get_cuda_device = preds.get_device()
                print("preds device: ", get_cuda_device)

            #get last value (prediction) and put in preds vector
            preds = torch.cat((preds, test_preds),0).cuda().to(device)
            #concatenate in all predictions tensor
            all_preds = torch.cat((all_preds, test_preds),0)

            #preds: tensor with predictions used to make more predictions. Its first dimension len gets till predictio range
            #test_inputs: tensor with ground truth values used to make predictions
            if (preds.shape[0]==prediction_range):
                preds = torch.tensor([]).cuda().to(device)
                test_inputs = torch.cat((test_inputs,test_all[i:i+prediction_range]),0)
            

    return all_preds

def treat_preds(all_preds, scaler, hostnames):
    preds_df_unscaled = pd.DataFrame(all_preds.cpu().numpy())
    #undo predictions rescaling
    preds_array_unscaled = scaler.inverse_transform(preds_df_unscaled)
    #replace predictions unscaled values in dataframe
    preds_df= pd.DataFrame(preds_array_unscaled, columns=hostnames)
    return preds_df

def save_metadata(dataset,
                PARAM,
                dtw_,
                mse_
                ):

    loop_name = 'My_loop'
    #save predictions metadata
    #write criterion in more readable manner
    if isinstance(settings.PARAM['criterion'], nn.MSELoss):
        crit = "MSE"
    elif isinstance(settings.PARAM['criterion'], SoftDTW):
        crit = "SoftDtw_gamma={}".format(settings.PARAM['criterion'].gamma)
    elif isinstance(settings.PARAM['criterion'], MAPE): 
        crit = "MAPE" 

    for h, dtw, mse in zip(dataset.columns, dtw_, mse_):
        with open('{}.txt'.format('./lstm/' + loop_name + "_global_var_and_metrics"),"a+") as f:
            f.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(
                    VERSION_NAME,
                    h,
                    settings.NUM_WEEKS,
                    settings.DB_QUERY_ARGS['start_date'],
                    crit,
                    settings.PARAM['seq_len'],
                    settings.PARAM['learning_rate'],
                    settings.PARAM['max_epochs'], 
                    settings.PARAM['num_layers'],                   
                    settings.PARAM['batch_size'],
                    settings.PARAM['hidden_size'],
                    settings.PARAM['dropout'],
                    settings.PARAM['duration_train_min'],
                    dtw,
                    mse
            ))
    print("metadata saved")



def do_all():
    #prepare device
    if torch.cuda.is_available and settings.GPU_DESIRED:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)

    #prepare dataset
    df_wks, _ = get_df()
    dataset, tensor_dataset, hostnames, scaler = df_to_treated_tensor(df_wks)
    train_all, test_all, valid_all = split_data(tensor_dataset)

    #Define network
    hidden_size = settings.PARAM["hidden_size"]
    encoder = EncoderLSTM(len(hostnames), hidden_size, settings.PARAM["num_layers"], settings.PARAM["dropout"]).to(device)
    decoder = BahdanauDecoder(hidden_size,len(hostnames), settings.PARAM["num_layers"], settings.PARAM["dropout"]).to(device)
    lr = settings.PARAM["learning_rate"]
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)

    epochs = 1
    criterion = settings.PARAM["criterion"]
    train_window = settings.PARAM["seq_len"]
    #train
    start_train = datetime.now()
    avg_loss = train(encoder, 
                decoder, 
                encoder_optimizer, 
                decoder_optimizer, 
                train_all, 
                train_window, 
                device, 
                hostnames,
                epochs,
                criterion)
    end_train = datetime.now()
    duration_train = (end_train - start_train).total_seconds()/60
    print("\n AVG LOSS \n", avg_loss)



    #PREDICT
    all_preds = test(encoder, 
                    decoder, 
                    settings.PARAM["seq_len"], 
                    train_all, 
                    test_all,
                    hostnames,
                    device)
    preds_df = treat_preds(all_preds, scaler, hostnames)

    #create directory if it doesnt exist
    direc = GLOBAL_DIREC + '{}/'.format(VERSION_NAME)
    check_folder = os.path.isdir(direc)
    # If folder doesn't exist, then create it.
    if not check_folder:
        os.makedirs(direc)

    #predictions error: error = dist_dtw_, mixed_err_
    dtw_, mse_ = compute_test_error(dataset, preds_df, len(train_all))
    #prediction html
    generate_predictions_html([dtw_, mse_], dataset, preds_df, len(train_all))
    print("predictions images saved")

    #save train and val loss
    fig64 = get_loss_img(avg_loss)
    im = Image.open(BytesIO(base64.b64decode(fig64)))
    im.save('{0}train_val_loss.png'.format(direc), 'PNG')
    print("loss images saved")

    #save metadata
    save_metadata(dataset,
                settings.PARAM,
                dtw_,
                mse_)
                


# Run as program
if __name__ == '__main__':
    do_all()