import torch
import torch.nn as nn
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
DATA_SIZE=200 #학습, 평가에 사용할 총 데이터 크기

Valid_Scale=500 #평가에 사용할 데이터 크기
MODEL='cstm'#학습에 사용할 모델
BATCH=1 #batch size, 한 번 학습할 때 들어가는 데이터 묶음
Learning_Rate=1e-4#gradient descent에서 한번에 어느정도 하강할지
EPOCH=3000 #학습 사이클을 도는 횟수


DATA_PATH='whole_data.xlsx' #학습 데이터 소스
Valid_Point = 100 #테스트데이터를 통해 학습 성능 평가하는 주기
CHECKPOINT_save=2000 #학습한 모델을 저장하는 주기
CHECKPOINT_dir='checkpoints' #학슴모델을 저장하는 위치
load_checkpoint='checkpoints/{}/{}_largestAccuracy.pth'
'''################################# 파라미터 설정 #################################'''

def test():
    seed = random.randint(1, 10000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if torch.cuda.is_available() : device = 'cuda'
    else : device = 'cpu'
    print('Device for training :',device)

    test_data=utility_tools.load_data(DATA_PATH,DATA_SELEC,DATA_SIZE,label_name)
    test_loader=utility_tools.transform_np_dataloader(test_data,PERIOD,BATCH,label_name)

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

            print('number : [{}] class : [{}]\n'.format(y_test_pred,torch.argmax(y_test_pred)))


if __name__ == "__main__":
    test()