import torch
import torch.nn as nn
import os
import numpy as np
import random
from model.cnn import CNN #딥러닝 모델 모듈 코드 임포트
from model.cstm import CSTM
import utility_tools

'''################################# 파라미터 설정 #################################'''
label_name='detail_label'
DATA_SELEC=[label_name,'bond3Y','KOSPI','KOSPI200','KOSDAQ','NASDAQ','NASDAQ100'] #데이터 선별
num_data=len(DATA_SELEC)-1
label_num=4
PERIOD = 80 #학습할 데이터의 기간 단위(일)
DATA_SIZE=120+PERIOD #투자 기간


MODEL='cstm'#학습에 사용할 모델
BATCH=1 #batch size, 한 번 학습할 때 들어가는 데이터 묶음

DATA_PATH='whole_data.xlsx' #학습 데이터 소스

load_checkpoint='checkpoints/{}/{}_largestAccuracy.pth'
'''################################# 파라미터 설정 #################################'''

def test():
    '''seed = random.randint(1, 10000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)'''


    if torch.cuda.is_available() : device = 'cuda'
    else : device = 'cpu'
    print('Device for training :',device)

    test_data=utility_tools.load_data(DATA_PATH,DATA_SELEC,DATA_SIZE,label_name,test=False)
    investing_data=utility_tools.load_data(DATA_PATH,DATA_SELEC,DATA_SIZE,label_name,test=True)
    test_loader=utility_tools.transform_np_dataloader(test_data,PERIOD,BATCH,label_name)

    count=0
    assets=0.0
    earning=0.0

    if MODEL=='cnn':
        model=CNN(batch_size=BATCH,input_size=num_data,label=label_num).to(device)
    elif MODEL=='cstm':
        model=CSTM(batch_size=BATCH,input_size=num_data).to(device)
    print('Testing model :',MODEL)
    model.load_state_dict(torch.load(load_checkpoint.format(MODEL,MODEL)))
    model.eval().to(device)
    criterion=torch.nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        for i,data in enumerate(test_loader):

            x_test=data[0].float()
            x_test=x_test.to(device)

            y_test_pred=model(x_test)


            #print(torch.argmax(y_test_pred))

            if torch.argmax(y_test_pred)==0: #little_buy
                count+=2
                assets-= 2*investing_data['bond3Y'][i+PERIOD]

            elif torch.argmax(y_test_pred) == 1:  # little_sell
                if count>0 :
                    assets += (count // 2) * investing_data['bond3Y'][i + PERIOD]
                    count -= (count//2)


            elif torch.argmax(y_test_pred) == 2: # very_buy
                count += 4
                assets -= 4 * investing_data['bond3Y'][i + PERIOD]

            elif torch.argmax(y_test_pred) == 3:  # very_sell
                if count > 0:
                    assets += count * investing_data['bond3Y'][i + PERIOD]
                    count = 0
                    earning+=assets
                    assets = 0
        print('bond count : {} assets : {} earnings : {}'.format(count,assets,earning))

if __name__ == "__main__":
    test()