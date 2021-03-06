import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import utility_tools #데이터전처리와 같은 함수들이 포함된 코드 임포트
from model.cnn import CNN #딥러닝 모델 모듈 코드 임포트
from model.cstm import CSTM
import random

'''################################# 파라미터 설정 #################################'''
label_name='detail_label'
DATA_SELEC=[label_name,'bond3Y','KOSPI','KOSDAQ','SP500','Gold'] #데이터 선별
num_data=len(DATA_SELEC)-1
label_num=4
PERIOD = 80 #학습할 데이터의 기간 단위(일)
DATA_SIZE=5000 #학습, 평가에 사용할 총 데이터 크기

Valid_Scale=500 #평가에 사용할 데이터 크기
MODEL='cnn'#학습에 사용할 모델
BATCH=32 #batch size, 한 번 학습할 때 들어가는 데이터 묶음
Learning_Rate=1e-4#gradient descent에서 한번에 어느정도 하강할지
EPOCH=1000 #학습 사이클을 도는 횟수


DATA_PATH='whole_data.xlsx' #학습 데이터 소스
Valid_Point = 100 #테스트데이터를 통해 학습 성능 평가하는 주기
CHECKPOINT_save=2000 #학습한 모델을 저장하는 주기
CHECKPOINT_dir='checkpoints' #학슴모델을 저장하는 위치
'''################################# 파라미터 설정 #################################'''
try:
    os.mkdir(CHECKPOINT_dir)
    os.mkdir(CHECKPOINT_dir + "/" + MODEL)
except:
    pass
def weight_init(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight.data)
def train():
    torch.manual_seed(12)
    torch.cuda.manual_seed(12)
    random.seed(12)
    np.random.seed(12)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # 이 코드 사용하면 처리속도 저하될 수 있음, 여차하면 주석처리 하고 실행

    if torch.cuda.is_available() : device = 'cuda'
    else : device = 'cpu'
    print('Device for training :',device)

    selected_data=utility_tools.load_data(DATA_PATH,DATA_SELEC,DATA_SIZE,label_name)
    train_loader, valid_loader=utility_tools.generate_dataset(selected_data,Valid_Scale,PERIOD,BATCH,label_name)

    if MODEL=='cnn':
        model=CNN(batch_size=BATCH,input_size=num_data,label=label_num).to(device)
    elif MODEL=='cstm':
        model=CSTM(batch_size=BATCH,input_size=num_data).to(device)
    print('Training model :',MODEL)


    #model.apply(weight_init)
    optimizer=optim.Adam(model.parameters(),lr=Learning_Rate)
    criterion=torch.nn.CrossEntropyLoss().to(device)

    largest_accuracy=0.0

    for epoch in range(EPOCH):

        average_cost=0.0
        model.train()
        for i,train_data in enumerate(train_loader):

            x_train=train_data[0].float()
            y_train=train_data[1].long()
            x_train=x_train.to(device)
            y_train=y_train.to(device)

            optimizer.zero_grad()

            y_train_pred=model(x_train)

            loss=criterion(y_train_pred,y_train)
            loss.backward()
            optimizer.step()

            average_cost+=loss/len(train_loader)
        if(epoch+1)%10==0:
            print('epoch : {:>04} loss = {:>.6}'.format(epoch+1,average_cost))


        if(epoch+1)%Valid_Point==0:
            print("\nvalidation testing...")
            model.eval()
            v_cost=0.0
            correct = 0

            with torch.no_grad():
                for j, valid_data in enumerate(valid_loader):

                    x_valid=valid_data[0].float()
                    y_valid=valid_data[1].long()
                    x_valid=x_valid.to(device)
                    y_valid=y_valid.to(device)

                    y_valid_pred=model(x_valid)

                    v_loss=criterion(y_valid_pred,y_valid)
                    prediction=y_valid_pred.max(1,keepdim=True)[1]
                    correct+=prediction.eq(y_valid.view_as(prediction)).sum().item()
                    v_cost+=v_loss/len(valid_loader)

            v_accuracy=100.*correct/len(valid_loader.dataset)
            print('\n[EPOCH : {:>4}]\t[Valid_loss : {:.6f}]\t[Valid Accuracy : {:.2f}]\n'.format(epoch+1,v_cost,v_accuracy))
            if v_accuracy>largest_accuracy:
                print('\nlargest accuracy [{:.2f}] detected!! Checkpoint Saving...\n'.format(v_accuracy))
                state_dict = model.state_dict()
                torch.save(state_dict, '{}/{}/{}_largestAccuracy.pth'.format(CHECKPOINT_dir, MODEL,MODEL))
                largest_accuracy=v_accuracy
    print('most large accuracy : {:.2f}'.format(largest_accuracy))







if __name__ == "__main__":
    train()