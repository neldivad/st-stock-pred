import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import plotly.express as px

start = '2015-01-01'
today= date.today().strftime('%Y-%m-%d')

st.title('Stock Prediction')

stocks = ('AAPL', 'GOOG', 'MSFT', 'GME', 'ADBE', 'AMD', 'PYPL')
    # Tuple object, not list, when used in st.selectbox()
selected_stocks = st.selectbox('Select ticker for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache
def load_data(ticker):
    data= yf.download(ticker, start, today)
    data.reset_index(inplace= True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stocks)
data_load_state.text('Loading data...done!')

st.subheader('Last 10 days')
st.write(data.tail(10) )

def plot_raw():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Open'],
        name='stock_open',
    ))
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Close'],
        name='stock_close',
    ))
    fig.layout.update(
        title_text= f'Time Series data for {selected_stocks}', 
        xaxis_rangeslider_visible=True
        )
    st.plotly_chart(fig)

plot_raw()

# Forecasting
df_train= data[['Date', 'Close']]
df_train= df_train.rename(
    columns= {
        'Date': 'ds', 
        'Close': 'y',
        })

m= Prophet()
m.fit(df_train)


future= m.make_future_dataframe(periods=period)
forecast= m.predict(future)

st.subheader('Forecast data')
st.write( forecast.tail() )

st.write('Forecast chart')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('Forecast component')
fig2 = m.plot_components(forecast)
st.write(fig2)
