import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class CSTM(nn.Module):
    def __init__(self,batch_size=1,input_size=4,hidden_size=2,num_layers=1):
        super(CSTM, self).__init__()
        self.batch_size = batch_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.input_size=input_size
        self.Convloution_1=nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=5, padding=2),#in_channel : 데이터 종류(예측 데이터 제외)
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
        self.Pool3 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.FCC = nn.Sequential(
            nn.Linear(5*512,256), #period가 pooling 3번 거쳐서 40->5가 됨
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            #nn.ReLU(),
            #nn.Linear(64, 2),
        )
        self.lstm=nn.Sequential(
            nn.LSTM(input_size=input_size,hidden_size=1,num_layers=2,batch_first=True),
            nn.ReLU()
        )
        self.FCL=nn.Sequential(
            nn.Linear(hidden_size,128),
            nn.ReLU()
        )
        self.FC_all=nn.Sequential(
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,16),
            nn.ReLU(),
            nn.Linear(16,2)
        )


    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to("cuda") # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to("cuda") # internal state
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)
        l_x=self.FCL(hn)

        c_x = self.Convloution_1(x)
        c_x = self.Pool(c_x)
        c_x = self.Convloution_2(c_x)
        c_x = self.Pool(c_x)
        c_x = self.Convloution_3(c_x)
        c_x = self.Pool(c_x)
        c_x = self.Convloution_4(c_x)
        c_x = self.Pool(c_x)
        c_x = c_x.view([self.batch_size, -1])
        c_x=self.FCC(c_x)

        out=torch.cat([c_x,l_x],dim=1)
        print(out.shape)

        c_x = F.log_softmax(c_x,dim=1)
        return x
if __name__=="__main__":
    model=CSTM(10).to("cuda")
    data_1=torch.ones([10,4,20]).to("cuda")
    pred_=model(data_1)
    print(model)
    print(pred_.shape)
