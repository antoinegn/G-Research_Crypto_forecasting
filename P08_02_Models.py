#!/usr/bin/env python
# coding: utf-8

# ## G-Research Cryptocurrencies : Models

# ### 1. Imports & Preprocessing

# In[1]:


# system libraries
import glob
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
import gc

# data manipulation libraries
import pandas as pd
import numpy as np

# graphical libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# modelisation libraries
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from keras.callbacks import ModelCheckpoint
from livelossplot import PlotLossesKeras
from statsmodels.tsa.seasonal import seasonal_decompose


# In[2]:


def regression_metrics(y_test, y_pred):
    """Function which contains differents metrics about regression
    Input: y_test, prediction
    
    Output: MAE, MSE, RMSE & MAPE 
    """
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print("MAE: ",mae.round(5))
    print("MSE: ", mse.round(5))
    print("RMSE: ", rmse)
    print("MAPE: ", mape.round(5))


# In[3]:


DATA_PATH = 'g-research-crypto-forecasting/prep'


# In[4]:


all_files = glob.glob(DATA_PATH + "/*.parquet.gzip")

li = []

for filename in all_files:
    df = pd.read_parquet(filename)
    li.append(df)

merged_df = pd.concat(li,
                  axis=0,
                  ignore_index=False)


# In[5]:


print(merged_df)


# In[6]:


del li, all_files
gc.collect()


# In[7]:


merged_df.fillna(0, inplace=True)


# In[8]:


merged_df.drop(["vwap", "count",
          "open", "close", "high",
          "low", "volume", "log_open",
          "log_close",
          "log_low", "log_high",
          "vwap", "count",
          "open", "close", "high",
          "low", "volume", "log_open",
          "log_close", "log_low",
          "log_high"],
          axis=1, inplace=True)


# In[9]:


merged_df = merged_df.sort_index()


# In[10]:


y = merged_df["target"].values
X = merged_df.drop("target", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.7,
                                                    random_state=42,
                                                    shuffle = False)


# In[11]:


pd.DataFrame(X_train).nunique()


# In[12]:


train_timestamp = X_train.index


# In[13]:


train_timestamp


# In[14]:


X_test


# In[15]:


y_train


# In[16]:


y_test


# In[17]:


del X,    y,    merged_df
gc.collect()


# In[18]:


idx = X_test.index


# In[19]:


cat_var = ["asset_name"]
num_var = ["H-L", "O-C", "MA_7d",
           "MA_14d", "MA_21d", "STD_7d"]


# In[20]:


cat_pipe = Pipeline([
    ('encoder', OneHotEncoder())
])

num_pipe = Pipeline([
    ('impute', SimpleImputer(strategy="mean")),
    ('scaler', StandardScaler())
])

preprocessing_pipe = ColumnTransformer(
    transformers=[
    ("cat", cat_pipe, cat_var),
    ('num', num_pipe, num_var)
])


# In[21]:


X_train = preprocessing_pipe.fit_transform(X_train)


# In[22]:


X_train


# ### 2. Modelisation

# #### 2.1 Linear Regression

# In[23]:


lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)


# In[24]:


X_test


# In[25]:


X_test = preprocessing_pipe.transform(X_test)


# In[26]:


y_pred = lin_reg.predict(X_test)


# In[27]:


regression_metrics(y_test, y_pred)


# In[28]:


pd.DataFrame(y_pred).index = idx


# In[29]:


# Visualizing the results for Linear reg
fig = go.Figure()
fig.add_trace(go.Scatter(x=pd.DataFrame(y_test).index, y=y_test,
                    mode='lines',
                    name='True'))
fig.add_trace(go.Scatter(x=pd.DataFrame(y_pred).index, y=y_pred,
                    mode='lines',
                    name='Predicted'))


# In[30]:


del y_pred, lin_reg
gc.collect()


# #### 2.2 Random Forest

# In[31]:


rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train, y_train)


# In[32]:


y_pred = rf_reg.predict(X_test)
regression_metrics(y_test, y_pred)


# In[33]:


# Visualizing the results for Linear reg
fig = go.Figure()
fig.add_trace(go.Scatter(x=pd.DataFrame(y_test).index, y=y_test,
                    mode='lines',
                    name='True'))
fig.add_trace(go.Scatter(x=pd.DataFrame(y_pred).index, y=y_pred,
                    mode='lines',
                    name='Predicted'))


# In[34]:


del y_pred,    rf_reg
gc.collect()


# #### 2.3 LSTM

# In[35]:


# Some functions to help out with
def plot_predictions(test,predicted):
    fig = px.line(x=test.index(), y="Close")
    #fig.add_trace(px.line(predicted))
    fig.show()

def return_rmse(test,predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {}.".format(rmse))


# In[36]:


y_train[60]


# In[37]:


# Since LSTMs store long term memory state, we create a data structure with 60 timesteps and 1 output
# So for each element of training set, we have 60 previous training set elements
history_points = 5
X_train_lstm = []
y_train_lstm = []
for i in range(history_points*14,X_train.shape[0]):
    X_train_lstm.append(X_train[i-history_points:i,0])
    y_train_lstm.append(y_train[i])
X_train_lstm, y_train_lstm = np.array(X_train_lstm), np.array(y_train_lstm)


# In[38]:


# The LSTM architecture
regressor = Sequential()

# First LSTM layer with Dropout regularisation
regressor.add(LSTM(units=32, return_sequences=True, input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.3))
# Fifth LSTM layer
regressor.add(LSTM(units=32, return_sequences=False))
regressor.add(Dropout(0.3))
# The output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam',loss='mean_squared_error')
filepath = "model.h5"
plotloss_cb = PlotLossesKeras()
checkpoint = ModelCheckpoint(filepath, monitor="loss", verbose=2, save_best_only=True, mode='min')
callbacks_list = [checkpoint, plotloss_cb]


# In[39]:


get_ipython().run_cell_magic('time', '', '\nhistory = regressor.fit(\n    X_train,\n    y_train,\n    shuffle = False,\n    callbacks = callbacks_list,\n    batch_size = 32,\n    validation_split = 0.2,\n    verbose = 1, epochs = 50)')


# In[40]:


y_pred = regressor.predict(X_test).flatten()


# In[41]:


regression_metrics(y_pred, y_test)


# In[42]:


# Visualizing the results for LSTM
fig = go.Figure()
fig.add_trace(go.Scatter(x=pd.DataFrame(y_test).index, y=y_test,
                    mode='lines',
                    name='True'))
fig.add_trace(go.Scatter(x=pd.DataFrame(y_pred).index, y=y_pred,
                    mode='lines',
                    name='Predicted'))


# In[43]:


error = y_pred - y_test
plt.hist(error, bins=100)
plt.xlabel('Prediction Error [USD]')
_ = plt.ylabel('Count')


# #### 2.4 Statistical Methods (SARIMA)

# ##### 2.4.1 Seasonal Decomposition

# In[44]:


X_train = pd.DataFrame(X_train)


# In[45]:


X_train.index = train_timestamp


# In[46]:


X_train


# In[47]:


X_train.resample('M').mean()


# In[121]:


#function to plot
plt.rcParams["figure.figsize"]=(15,7)

def season_df(data,label,time):
    df = data.resample(time).mean()
    
    seasonal_decompose(df[15]).plot()
    print(label)
    return plt.show()

season_df(X_train, "Seasonal Decomposition", 'M')
season_df(X_train, "Seasonal Decomposition", 'D')
season_df(X_train, "Seasonal Decomposition", 'H')


# In[84]:


from scipy import stats
from itertools import product
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[122]:


hourly = X_train.resample('H').mean()
daily = X_train.resample('D').mean()
monthly = X_train.resample('M').mean()


# In[123]:


hourly.head(5)


# In[124]:


hourly = hourly.rename(columns={15: "O-C"})
daily = daily.rename(columns={15: "O-C"})
monthly = monthly.rename(columns={15: "O-C"})


# In[125]:


def sarimax_function(df):
    qs = range(0, 3)
    ps = range(0, 3)
    d=1
    parameters = product(ps, qs)
    parameters_list = list(parameters)
    len(parameters_list)

    # Model Selection
    results = []
    best_aic = float("inf")
    warnings.filterwarnings('ignore')
    for param in parameters_list:
        try:
            model = SARIMAX(df['O-C'], order=(param[0], d, param[1])).fit(disp=-1)
        except ValueError:
            print('bad parameter combination:', param)
            continue
        aic = model.aic
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])

    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    print(result_table.sort_values(by = 'aic', ascending=True).head())
    print(best_model.summary())
    return best_model
    


# In[128]:


hrmodel = sarimax_function(hourly)
hrmodel.plot_diagnostics(figsize=(15, 12))
plt.show()


# In[127]:


dymodel = sarimax_function(daily)
dymodel.plot_diagnostics(figsize=(15, 12))
plt.show()


# In[129]:


mtmodel = sarimax_function(monthly)
mtmodel.plot_diagnostics(figsize=(15, 12))
plt.show()


# In[192]:


y_true_sarimax = y_test.copy()
y_true_sarimax = pd.DataFrame(y_true_sarimax)
y_true_sarimax.index = idx
y_true_sarimax = y_true_sarimax.resample('H').mean()


# In[193]:


y_true_sarimax


# In[196]:


y_pred_sarimax = hrmodel.predict(start = '2020-08-30 23:00:00', end = '2021-09-20 23:00:00', dynamic = True)


# In[197]:


y_pred_sarimax


# In[198]:


regression_metrics(y_pred_sarimax, y_true_sarimax)


# In[199]:


# Visualizing the results for Linear reg
fig = go.Figure()
fig.add_trace(go.Scatter(x=pd.DataFrame(y_pred_sarimax).index, y=y_true_sarimax[0],
                    mode='lines',
                    name='True'))
fig.add_trace(go.Scatter(x=pd.DataFrame(y_pred_sarimax).index, y=y_pred_sarimax,
                    mode='lines',
                    name='Predicted'))

