# AI 기반 주가예측 프로그램

-------

## 1. 개요

본 프로젝트는 AI 전문가 교육과정의 일환으로 파일럿프로젝트로 수행하였다.
AI 기반 주가예측 프로그램 구현을 위해 위키독스의 [금융 데이터 분석을 위한 파이썬](https://wikidocs.net/173005) 문서를 기반으로 작성하였고, 이를 개선시키는 방향으로 처리하였다.

![실행화면](images/screenshot_running.png)
[시연영상](https://youtu.be/u30TsraUVXs)

### 1.1. 개선사항

- Deep Learing을 위해 사용하는 데이터를 주가시세 데이터 외에 종목기초정보와 환율데이터를 사용하도록 추가
- tensorflow를 사용했던 코드를 pytorch를 사용하도록 수정
- streamlit을 사용하여 시각화
- 사용자가 변경할 수 있는 hyper-parameter를 수정가능하도록 처리

### 1.2. 사용 모델 및 라이브러리

- 사용모델 : LSTM (시계열예측에 특화된 Deep Learning Model)
- 주식시세 및 환율정보 수집 : [pykrx](https://github.com/sharebook-kr/pykrx), [yfinance](https://pypi.org/project/yfinance/)
- Deep Learning Library : pytorch
- 데이터 전처리 및 모델평가 : scikit-learn, pandas, numpy
- 시각화 및 사용자 UI : matplotlib, streamlit

## 2. 수행내용

### 2.1. 데이터 수집

- 주식시세 데이터와 종목 기초정보, 환율 정보를 각각 입수

### 2.2. 데이터 전처리 

- 입수한 데이터를 날짜를 기준으로 Dataframe Merge 실행하고 MinMaxScaler를 이용하여 데이터 스케일링 처리
- 학습데이터와 테스트데이터 준비(예측을 위한 학습데이터를 D+1 day의 주가 학습을 위해 D-10 ~ D day까지 10일간의 데이터를 사용하도록 구성)
- train_test_split 함수를 통해 학습데이터와 테스트데이터 분리

### 2.3. 모델학습 및 평가

- Hyper-Parameter 값을 조정하면서 모델학습 수행
- 모델학습 결과를 MSE와 MAE를 통해 평가

## 3. 결론
1. 수집된 데이터가 많을 수록 실데이터에 수렴
2. 학습횟수에 따른 모델학습 평가점수가 좋아지나, 일정 횟수를 넘으면 과적합 등의 이유로 오히려 안좋아지는 결과 확인
3. 전반적인 추세선은 모방하지만, 선반영은 안되는 문제 확인

## 4. 개선방안
1. 뉴스기사 및 경제동향 데이터 수집을 추가하여 텍스트 분석 등을 통한 주가예측 기능추가(주가추세 선반영 등 기능개선)
2. 의사결정 모델을 적용하여 매수/매도 타이밍 제시



