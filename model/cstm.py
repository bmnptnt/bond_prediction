import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CSTM(nn.Module):
    def __init__(self,batch_size=1,input_size=4,label=4):
        super(CSTM, self).__init__()
        self.batch_size = batch_size
        self.hidden_size=64
        self.Convloution_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=5, padding=2),
            # in_channel : 데이터 종류(예측 데이터 제외)
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        self.Convloution_2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )

        self.Convloution_3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        self.Convloution_4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        self.Pool = nn.MaxPool1d(kernel_size=2,stride=2)

        self.lstm = nn.LSTM(input_size=256, hidden_size=self.hidden_size, num_layers=1, batch_first=True)

        self.FC = nn.Sequential(
            nn.Linear(self.hidden_size*10,80), #period가 pooling 두번 거쳐서 20->5가 됨
            #nn.BatchNorm1d(80),
            nn.ReLU(),
            nn.Linear(80, 20),
            #nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Linear(20, label),
        )

    def forward(self, x):


        x = self.Convloution_1(x)
        x = self.Pool(x)
        x = self.Convloution_2(x)
        x = self.Pool(x)
        x = self.Convloution_3(x)
        x = self.Pool(x)
        #x = self.Convloution_4(x)
        #x = self.Pool(x)
        #x = self.Convloution_3(x)
        #x = self.Pool(x)

        x = x.permute((0, 2, 1))

        h_0 = Variable(torch.zeros(1, x.size(0), self.hidden_size)).to('cuda')
        c_0 = Variable(torch.zeros(1, x.size(0), self.hidden_size)).to('cuda')
        #print(x.shape)
        output, (hn, cn) = self.lstm(x, (h_0, c_0))

        #print(output.shape)

        x = output.contiguous().view([self.batch_size, -1])
        #print(x.shape)
        x=self.FC(x)

        x = F.log_softmax(x,dim=1)

        return x

if __name__=="__main__":
    model=CSTM(10).to("cuda")
    data_1=torch.ones([10,4,20]).to("cuda")
    pred_=model(data_1)
    print(model)
    print(pred_.shape)