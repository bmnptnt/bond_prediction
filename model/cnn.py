import torch
import torch.nn as nn
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self,batch_size=1,numOFdata=4):
        super(CNN, self).__init__()
        self.batch_size = batch_size

        self.Convloution_1=nn.Sequential(
            nn.Conv1d(in_channels=numOFdata, out_channels=64, kernel_size=5, padding=2),#in_channel : 데이터 종류(예측 데이터 제외)
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
        self.FC = nn.Sequential(
            nn.Linear(5*512,256), #period가 pooling 3번 거쳐서 40->5가 됨
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.Convloution_1(x)
        x = self.Pool(x)
        x = self.Convloution_2(x)
        x = self.Pool(x)
        x = self.Convloution_3(x)
        x = self.Pool(x)
        x = self.Convloution_4(x)
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
