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
DATA_SELEC=[label_name,'bond3Y','KOSPI','SP500','Gold','KR_CPI','KR_CA','KR_IMM','KR_Export'] #데이터 선별
num_data=len(DATA_SELEC)-1
label_num=4
PERIOD = 80 #학습할 데이터의 기간 단위(일)
DATA_SIZE=120+PERIOD #투자 기간


MODEL='cnn'#학습에 사용할 모델
BATCH=1 #batch size, 한 번 학습할 때 들어가는 데이터 묶음

DATA_PATH='whole_data.xlsx' #학습 데이터 소스

load_checkpoint='checkpoints/{}/{}_largestAccuracy.pth'
'''################################# 파라미터 설정 #################################'''

def test():
    torch.manual_seed(12)
    torch.cuda.manual_seed(12)
    random.seed(12)
    np.random.seed(12)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True #이 코드 사용하면 처리속도 저하될 수 있음, 여차하면 주석처리 하고 실행


    if torch.cuda.is_available() : device = 'cuda'
    else : device = 'cpu'
    print('Device for training :',device)

    test_data=utility_tools.load_data(DATA_PATH,DATA_SELEC,DATA_SIZE,label_name,test=False)
    investing_data=utility_tools.load_data(DATA_PATH,DATA_SELEC,DATA_SIZE,label_name,test=True)
    test_loader=utility_tools.transform_np_dataloader(test_data,PERIOD,BATCH,label_name)

    count=0
    assets=10000000

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




            if torch.argmax(y_test_pred)==0: #little_buy
                count+=(assets//10000)//10
                assets-= ((assets//10000)//10)*(investing_data['bond3Y'][i+PERIOD]*10000)

            elif torch.argmax(y_test_pred) == 1:  # little_sell
                if count>0 :
                    assets += (count // 2) * (investing_data['bond3Y'][i + PERIOD]*10000)
                    count -= (count//2)


            elif torch.argmax(y_test_pred) == 2: # very_buy
                count += (assets // 10000) // 5
                assets -= ((assets // 10000) // 5) * (investing_data['bond3Y'][i + PERIOD] * 10000)

            elif torch.argmax(y_test_pred) == 3:  # very_sell
                if count > 0:
                    assets += count * (investing_data['bond3Y'][i + PERIOD] * 10000)
                    count = 0

        print('bond count : {} assets : {}'.format(count,assets))

if __name__ == "__main__":
    test()