# Bond Prediction
- Class : 세종대학교 2022년 1학기 캡스톤(데이터사이언스)
- Topic : 채권 시장의 데이터를 이용한 채권 주요 지표 분석 및 트레이딩 전략 구축
- Term : 2022/03 ~ 2022/06


## 데이터셋
- ['2001-01-02'] - ['2021-07-12'] 간의 경제지표 (※중간에 공백인 지표 존재)

## 전략구축방법
-Deep learning 모델에 각종 경제지표를 학습시켜 채권의 양상을 예측 후 최종적인 투자전략 구축-
#### <del>채권 가격 예측</del>(폐기된 방법) 
- <del>LSTM</del>
#### 채권 양상 분류 및 투자 전략 설정(최종 선정)
- CNN : VGG16의 일부 블록 (1d convolution 블록 활용)
- CSTM : CNN (VGG16의 일부 블록, 1d conv)을 앞에 배치하고 뒷 부분에 LSTM을 배치 
##### 분류 기준 
- 2 Labels : 향후 20일(한 달) 간의 채권 수익률 평균이 현재보다 높을 때 buy, 낮을 때 sell
- 4 Labels : 20일간 평균이 현재보다 높으며 높은 일수가 12일 이상일 때 very buy/ 높은 일수가 12일 미만일 때 little buy/ 20일간 평균이 현재보다 낮으며 낮은 일수가 8일 이상일 때 little sell/ 낮은 일수가 8일 미만일 때 very sell

| Week | Goal | Method | Result | Feedback | Note | 
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 주제 선정 |  | 채권전략분석및구축으로 주제 결정 | 채권과 그에 따른 경제지표 이해 필요 | |
| 2~4 | 채권 관련 지식 학습 | 데이터와 그에 따른 방법론 탐색, seaborn으로 데이터간의 Correlation파악 | Correlation에 따른 지표 선별 | 다중공선성 문제 | |
| 5~7 | 미래의 채권 가격 예측 | LSTM을 활용하여 시계열 데이터인 채권 가격 예측 | 과거의 데이터를 따라가는 cheating발생 | cheating문제, 가격 예측과 주제의 적합성 | |
| 8~10 | 채권 양상 분류 | CNN(1d)을 활용하여 데이터분석 후 투자형태 분류 | 약 30~60%의 정확도 |  | |
| 11~13 | 채권 양상 분류 | CNN(1d)과 LSTM 합친 CSTM을 활용하여 데이터분석 후 투자형태 분류 |  | 분류에 따른 모의투자를 통해 수익확인 필요 | |
| 14~15 | 모의투자, 최종시연 및 발표 | CSTM을 활용하여 데이터분석 후 분류 및 모의투자 |  |  | 10,000,000원의 모의투자 예산 책정 |
## References
- 파이썬 딥러닝 파이토치 (2020)

