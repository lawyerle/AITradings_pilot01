import datetime
import streamlit as st

from pykrx import stock
from pykrx import bond
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error

# CSS to adjust the width of the sidebar and main content
st.markdown(
    """
    <style>
    /* Sidebar width */
    [data-testid="stSidebar"] {
        width: 500px;
    }

    /* Main content width */
    .css-1d391kg {
        max-width: 1000px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

company = ''
end_date = ''
start_date = ''

# HyperParameter
window_size = 10
test_size = 0.28
random_state = 21
dropout=0.3
epochs = 70
batch_size = 30

features = ['시가', '고가', '저가', '종가'] 
feature_set = []
target = ['종가']

now = datetime.datetime.now()
df_company = None
df_ohlcv = None
dfy = None
learning_period = ""

# ticker_name = stock.get_index_ticker_name(company)
def calc_start_end_date():
    global start_date, end_date
    if learning_period == '1년':
        calc_date = now - relativedelta(years=1)
    elif learning_period == '3년':
        calc_date = now - relativedelta(years=3)
    elif learning_period == '5년':
        calc_date = now - relativedelta(years=5)
    elif learning_period == '7년':
        calc_date = now - relativedelta(years=7)
    else:
        calc_date = now - relativedelta(years=10)
        
    end_date = now.strftime('%Y-%m-%d')
    start_date = calc_date.strftime('%Y-%m-%d')
    

def get_company_code(company_name): 
    # 나중에 세션에 정보 저장
    global df_company 
    # ticker 리스트를 가져오지 않았으면 티커리스트 DataFrame을 생성
    if df_company == None :
        df_company = pd.DataFrame()
        
        str_today = now.strftime('%Y%m%d')
        
        market_list = ['KOSPI', 'KOSDAQ', 'KONEX']
        
        for market in market_list:
            ticker_list = stock.get_market_ticker_list(str_today, market=market)
            
            for ticker in ticker_list : 
                # print(ticker)
                ticker_name = stock.get_market_ticker_name(ticker)
                # print(ticker_name)
                
                df = pd.DataFrame({'ticker': ticker,
                                'corp_name': ticker_name, 
                                'market': market
                                }, index = [0])
                
                df_company  = pd.concat([df_company, df])
            
        df_company = df_company.reset_index(drop=True)
        
    # print(df_company.head())
    
    return df_company.loc[df_company['corp_name'] == company_name, 'ticker'].iloc[0]
    

def get_market_data(code) :
    global df_ohlcv, start_date, end_date
    print(start_date)
    print(end_date)
      
    #주가의 OHLCV 값을 조회
    df_ohlcv = stock.get_market_ohlcv(start_date, end_date, code)
    print(df_ohlcv.head())
    #일자별 DIV/BPS/PER/EPS를 조회
    #DIV : 배당
    #BPS : 주당 순자산가치
    #PER : 주당 순이익비율
    #EPS : 주당 순이익
    df_dbpe = stock.get_market_fundamental(start_date, end_date, code)

    # 두개의 dataframe join merge
    df_result = pd.merge(df_ohlcv, df_dbpe, how='inner', on='날짜')

    # 환율 데이터 조회
    df_exchange = yf.download('KRW=X', start=start_date, end=end_date)

    # index field(날짜)의 명칭을 맞추기 위한 컬럼명 변경작업
    df_exchange.reset_index(inplace=True)
    df_exchange = df_exchange[['Date', 'Close']]

    df_exchange.rename(columns = {'Date' : '날짜', 'Close' : '환율종가'}, inplace = True)
    df_exchange.set_index('날짜', inplace=True)

    # 기존 데이터프레임과 환율 정보를 가진 데이터프레임을 merge
    df_result = pd.merge(df_result, df_exchange, how='inner', on='날짜')
    
    return df_result


def data_preprocess(df_data) :
    global dfy
    global dfx
    global scaler
    # 데이터 전처리 작업 수행
    scaler = MinMaxScaler()

    # print(featuer_set)
    features.extend(feature_set)
    # features.append('BPS')
    # features.append('PER')
    # features.append('환율종가')

    dfx = df_data[features]
    scaled_data = scaler.fit_transform(dfx)
    # dfx = pd.DataFrame(scaled_data, columns=dfx.columns)
    dfy = dfx[target]
    dfx = dfx[features]

    X = dfx.values.tolist()
    y = dfy.values.tolist()

    #window_size에 지정된 일수을 제외한 나머지 데이트를 학습시킬 데이터로 준비
    data_X = []
    data_y = []

    for i in range(len(y) - window_size) :
        _X = X[i : i + window_size] # [0:10], [1:11], [2:12] ....
        _y = y[i + window_size] # [10], [11], [12] ...
        data_X.append(_X)
        data_y.append(_y)


    # train data와 test 데이터 생성
    #시계열데이터이라 shuffle되면 안됨
    train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, shuffle=False, test_size=test_size, random_state=random_state)

    train_X = np.array(train_X)
    test_X = np.array(test_X)
    train_y = np.array(train_y)
    test_y = np.array(test_y)
    
    return train_X, test_X, train_y, test_y


class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=len(features), hidden_size=20, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(input_size=20, hidden_size=20, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(20, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)  # LSTM layer 1
        x, _ = self.lstm2(x)  # LSTM layer 2
        x = x[:, -1, :]  # 마지막 타임스텝의 출력을 가져옴
        x = self.fc(x)  # Fully connected layer
        return x

def train_model(train_X, train_y, test_X, test_y) :
    # 모델 인스턴스 생성
    model = LSTMModel()

    criterion = nn.MSELoss()  # 평균 제곱 오차 손실 함수
    optimizer = optim.Adam(model.parameters())  # Adam 옵티마이저

    # 모델 학습
    model.train()  # 훈련 모드로 전환
    for epoch in range(epochs):
        for i in range(0, len(train_X), batch_size):
            # 배치 데이터 가져오기
            batch_X = torch.tensor(train_X[i:i + batch_size], dtype=torch.float32)
            batch_y = torch.tensor(train_y[i:i + batch_size], dtype=torch.float32)

            # 경량화 이전 초기화
            optimizer.zero_grad()
            # 순전파
            outputs = model(batch_X)
            
            # 손실 계산
            loss = criterion(outputs, batch_y)

            # 역전파 및 가중치 업데이트
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    # 모델 평가
    model.eval()  # 평가 모드로 전환
    with torch.no_grad():
        test_X_tensor = torch.tensor(test_X, dtype=torch.float32)
        pred_y = model(test_X_tensor)
        test_y_tensor = torch.tensor(test_y, dtype=torch.float32)
        loss = criterion(pred_y, test_y_tensor)

    # print('loss : ', loss.item())
    mse = mean_squared_error(test_y, pred_y)
    mae = mean_absolute_error(test_y, pred_y)
    
    st.markdown(f"""
                **모델평가결과**
                1. Mean Squared Error : {mse}
                2. Mean Absolute Error : {mae}
                """)
    
    predict_df = dfx[features][-len(test_y):]
    
    numpy_array = pred_y.numpy()
    predict_df['종가'][:] = np.squeeze(numpy_array)
    predict_df[features] = scaler.inverse_transform(predict_df)
    predict_df = predict_df.applymap(lambda x : int(x))

    recent_df = df_ohlcv[features][-len(test_y):]
    
    return predict_df, recent_df

    # model.eval()  # 평가 모드로 전환
    # with torch.no_grad():
    #     pred_y = model(torch.tensor(test_X, dtype=torch.float32))

def draw_graph(pred_y, test_y):
    plt.rcParams.update({'font.size': 25})
    plt.figure(figsize=(25,15))
    plt.plot(test_y['종가'], color='red', label=f'{company_name} 실제 주가')
    plt.plot(pred_y['종가'], color='blue', label=f'{company_name} 예측 주가')
    plt.title(f'{company_name} 주가 예측 그래프')
    plt.xlabel('date')
    plt.ylabel('stock price')
    plt.legend()
    
    return plt

def write_estimate(pred_y):
    st.write(f"오늘 {company_name} 종가 :", df_ohlcv['종가'][-1], 'KRW')
    st.write(f"내일 {company_name} 주가(예측) :", pred_y['종가'][-1], 'KRW')
    sub_calc = pred_y['종가'][-1] - df_ohlcv['종가'][-1]
    st.write("증시 예상", '(상승)' if sub_calc >= 0  else '(하락)', ':', sub_calc, 'KRW' )
    
if __name__ == '__main__' :
    st.title("주식가격 예측 프로그램")
    st.sidebar.title('설정')
    
    try:
        company_name = st.sidebar.text_input("예측할 종목명을 입력하세요.")
    except IndexError:
        st.write("종목을 찾을 수가 없습니다.")
        
    learning_period = st.sidebar.radio("학습기간 : ", ["1년", "3년", "5년", "7년", "10년"])
    st.sidebar.subheader("Hyper Parameter: ")
    epochs = st.sidebar.slider("epoch 횟수 선택:", 10, 100, 70)
    batch_size = st.sidebar.slider("batch 사이즈 선택:", 10, 50, 30)
    dropout = st.sidebar.number_input("dropout 값 입력:", value=0.3, min_value=0.0, max_value=0.5, step=0.1, format="%.1f")
    featuer_set = st.sidebar.multiselect("Feature 선택:", ["DIV", "BPS", "PER", "EPS", "환율종가"])

    try:
        if st.sidebar.button("조회") and company_name :  
            calc_start_end_date() 
            company_code = get_company_code(company_name)
            st.header(f'{company_name}({company_code}) 주가예측 결과')
            df_data = get_market_data(company_code)
            train_X, test_X, train_y, test_y = data_preprocess(df_data)
            pred_y, test_y = train_model(train_X, train_y, test_X, test_y)
            st.pyplot(draw_graph(pred_y, test_y))
            write_estimate(pred_y)    
    except IndexError:
        st.write("종목을 찾을 수가 없습니다.")
    