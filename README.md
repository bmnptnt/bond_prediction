# Bond Prediction
- Class : 세종대학교 2022년 1학기 캡스톤(데이터사이언스)
- Topic : 채권 시장의 데이터를 이용한 채권 주요 지표 분석 및 트레이딩 전략 구축
- Term : 2022/03 ~ 2022/06


## 데이터셋
- ['2001-01-02'] - ['2021-07-12'] 간의 경제지표 (※중간에 공백인 지표 존재)

## 전략구축방법
-Deep learning 모델에 각종 경제지표를 학습시켜 채권의 양상을 예측 후 최종적인 투자전략 구축-
#### 채권 가격 예측 
- LSTM
#### *채권 양상 분류 및 투자 전략 설정(최종 선정)
- CNN : VGG16의 일부 블록 (1d convolution 블록 활용)
- CSTM : CNN (VGG16의 일부 블록, 1d conv)을 앞에 배치하고 뒷 부분에 LSTM을 배치 
| Week | Goal | Method | Result | Feedback | Note | 
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 주제 선정 |  | 채권전략분석및구축으로 주제 결정 | 채권과 그에 따른 경제지표 이해 필요 | |
| 2~3 | 채권 관련 지식 학습 | 데이터와 그에 따른 방법론 탐색, seaborn으로 데이터간의 Correlation파악 | Correlation에 따른 지표 선별 | 다중공선성 문제 | |
| 4~6 | 미래의 채권 가격 예측 | LSTM을 활용하여 시계열 데이터인 채권 가격 예측 | 과거의 데이터를 따라가는 cheating발생 | cheating문제, 가격 예측과 주제의 적합성 | |
| 8~10 | 채권 양상 분류 | CNN(1d)을 활용하여 데이터분석 후 투자 방법 분류 | 약 30~60%의 정확도 |  | |
| 11~15 |  | |  |  | |
## References
- 파이썬 딥러닝 파이토치 (2020)

