#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import datetime as dt
from datetime import datetime
import time
from datetime import datetime, date, time, timedelta


# In[41]:


from sklearn.linear_model import LinearRegression


# In[42]:


natgas_df = pd.read_csv("C:/Users/Lakshmi.C/Desktop/2024/forage/jpmorgan/Nat_Gas.csv")


# In[43]:


natgas_df


# In[44]:


plt.plot(natgas_df['Dates'], natgas_df['Prices'])


# In[45]:


#select only January Prices
natgas_df['Dates'] = pd.to_datetime(natgas_df['Dates'])
natgas_df['Year'] =natgas_df['Dates'].dt.year
natgas_df['Month'] =natgas_df['Dates'].dt.month

#Select Jan months
natgas_jan = natgas_df[natgas_df['Month'] == 1]
natgas_jan


# In[46]:


from sklearn.linear_model import LinearRegression
x = np.array(natgas_df[natgas_df['Month'] == 1]['Year'] ).reshape(-1,1)
y = np.array(natgas_df[natgas_df['Month'] == 1]['Prices'])
reg = LinearRegression().fit(x,y)
#reg


# In[47]:


round(float(reg.predict([[2025]])), 2)


# In[48]:


from sklearn.linear_model import LinearRegression

def next_year_price(next_year):
    """Returns a predicted natural gas prices for each month in the following year"""
    price_list = []
    for i in np.arange(12):
        X = np.array(natgas_df[natgas_df['Month'] == 1+i]['Year'] ).reshape(-1,1)
        y = np.array(natgas_df[natgas_df['Month'] == 1+i]['Prices'])
        reg = LinearRegression().fit(X,y)
        price = reg.predict([[next_year]])
        price_list.append(round(float(price), 2))
        
    return price_list


# In[49]:


gas_price25 = next_year_price(2025)
np.array(gas_price25)


# In[50]:


def get_last_of_each_month(year):
    dates_array = []
    current_date = datetime(year,12,31)  #start from the last day of the year
    while current_date.year == year:
        
        dates_array.append(current_date.strftime('%Y-%m-%d'))
        month = current_date.month                  
        year  = current_date.year
        
        #Move to the first  day of the previous month
        current_date = current_date.replace(year=year, month=month, day=1)
        
        # Move back one day to get the last day of current month
        current_date -= timedelta(days=1)
        
    return dates_array[::-1] #Reverse the array to get the dates in ascending order


# In[51]:


dates_2025 = get_last_of_each_month(2025)
dates_2025


# In[54]:


# New dataframe of 2025 dates and prices
projected_gas_prices25_df = pd.DataFrame({'Dates' : dates_2025, 'Prices' : gas_price25})
projected_gas_prices25_df
projected_gas_prices25_df['Dates'] = pd.to_datetime(projected_gas_prices25_df['Dates'])
projected_gas_prices25_df['Year'] = projected_gas_prices25_df['Dates'].dt.year
projected_gas_prices25_df['Month'] = projected_gas_prices25_df['Dates'].dt.month
projected_gas_prices25_df


# In[55]:


gas_df = pd.concat([natgas_df, projected_gas_prices25_df], ignore_index =True)
gas_df


# In[57]:


def get_gas_price(month, year):
    ''' Input month and year to retrieve predicted or historic gas price'''
    print(gas_df[(gas_df['Year'] == year) & (gas_df['Month'] == month)] ['Prices'])


# In[59]:


get_gas_price(10, 2025)


# In[66]:


# Final analysis 
plt.plot(gas_df['Dates'], gas_df['Prices'], label = 'Predicted 2025')
plt.plot(natgas_df['Dates'], natgas_df['Prices'], label = 'Actuals 2021-24')
plt.ylabel('Gas Price ')
plt.xlabel ('Year')
plt.title('Gas Price Forcast', fontweight = 'bold')
plt.legend()


# In[ ]:





# In[ ]:




