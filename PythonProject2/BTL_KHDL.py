import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.title("📈 Dự báo giá cổ phiếu bằng LSTM")
st.write("Tải lên một file CSV chứa dữ liệu giá cổ phiếu để phân tích và dự báo.")

uploaded_file = st.file_uploader("📤 Vui lòng tải lên file CSV có cột 'Date' và 'Close'.", type="csv")

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Kiểm tra cột cần thiết
    if 'Date' not in df.columns or 'Close' not in df.columns:
        st.error("❌ File CSV phải có cột 'Date' và 'Close'.")
    else:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        st.success("✅ File hợp lệ. Hiển thị dữ liệu gần nhất:")
        st.dataframe(df.tail(10))

        # Chuẩn bị dữ liệu cho LSTM
        data = df[['Close']].values.astype('float32')

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)

        look_back = 10  # số bước thời gian dựa vào để dự đoán

        X, Y = create_dataset(data_scaled, look_back)

        # Chia tập train/test: 80% train
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        Y_train, Y_test = Y[:train_size], Y[train_size:]

        # Reshape input cho LSTM [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Xây dựng mô hình LSTM
        model = Sequential()
        model.add(LSTM(50, input_shape=(look_back, 1)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Huấn luyện mô hình
        st.write("⏳ Đang huấn luyện mô hình LSTM...")
        model.fit(X_train, Y_train, epochs=20, batch_size=16, verbose=0)
        st.success("✅ Đã huấn luyện xong mô hình!")

        # Dự báo
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Chuyển ngược về giá trị thực
        train_predict = scaler.inverse_transform(train_predict)
        Y_train_real = scaler.inverse_transform(Y_train.reshape(-1,1))
        test_predict = scaler.inverse_transform(test_predict)
        Y_test_real = scaler.inverse_transform(Y_test.reshape(-1,1))

        # Vẽ biểu đồ so sánh dự báo
        st.subheader("Biểu đồ dự báo giá cổ phiếu")

        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(df['Date'][look_back:look_back+len(train_predict)], train_predict, label='Dự báo train')
        ax.plot(df['Date'][look_back+len(train_predict):look_back+len(train_predict)+len(test_predict)], test_predict, label='Dự báo test')
        ax.plot(df['Date'], df['Close'], label='Giá thực tế', color='black', alpha=0.6)
        ax.set_xlabel('Ngày')
        ax.set_ylabel('Giá đóng cửa')
        ax.legend()
        st.pyplot(fig)
else:
    st.info("📥 Vui lòng tải lên file CSV để bắt đầu dự báo.")
