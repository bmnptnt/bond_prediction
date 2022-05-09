import torch
import torch.nn as nn
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self,batch_size=1,numOFdata=4):
        super(CNN, self).__init__()
        self.batch_size = batch_size

        self.Convloution_1=nn.Sequential(
            nn.Conv1d(in_channels=numOFdata, out_channels=64, kernel_size=3, padding=1),#in_channel : 데이터 종류(예측 데이터 제외)
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.Convloution_2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.Pool = nn.MaxPool1d(kernel_size=2,stride=2)

        self.FC = nn.Sequential(
            nn.Linear(5*128,80), #period가 pooling 두번 거쳐서 20->5가 됨
            nn.BatchNorm1d(80),
            nn.ReLU(),
            nn.Linear(80, 20),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Linear(20, 2),
        )

    def forward(self, x):
        x = self.Convloution_1(x)
        x = self.Pool(x)
        x = self.Convloution_2(x)
        x = self.Pool(x)

        #x = x.view(-1, 5*128)
        x = x.view([self.batch_size, -1])
        x=self.FC(x)

        x = F.log_softmax(x,dim=1)
        return x
if __name__=="__main__":
    model=CNN(10).to("cuda")
    data_1=torch.ones([10,4,20]).to("cuda")
    pred_=model(data_1)
    print(model)
    print(pred_.shape)
