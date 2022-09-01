#!/usr/bin/env python
# coding: utf-8

# # Case Study - Portfolio Optimization
# In this case study, we will build two $10,000 investment portfolios containing four stocks. The first portfolio will have an equal weighting between the stocks. The second portfolio will be optimized with a weighting allocation that provides the best return, adjusted for risk. To build these two portfolios, we will:
# 1. Import two years of data for four stocks
# 2. Build the initial portfolio with equal weighting to each of the stocks
# 3. Analyze and visualize the equal-weighted portfolio
# 4. Generate 10,000 portfolio scenarios with random weighting to each of the stocks
# 5. Identify the optimal portfolio from the scenarios and visualize the results

# ## Import Packages & Connect to Data

# In[1]:


# Import packages needed for case study
import pandas as pd
import numpy as np
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# Set start and end date parameters
startdate = '2019-01-01'
enddate = '2021-12-31'

# List the four stock ticker symbols for our portfolio
stock_list = ['AMD', 'AAPL', 'MSFT', 'ORCL']

# Create an empty dictionary to store our stock info
stocks = {}

# Loop through each stock in the stock_list and return the Adj Close
for i_stock in stock_list:
    stocks[i_stock] = pdr.DataReader(i_stock, 'yahoo', startdate, enddate)[['Adj Close']]


# In[6]:


# Examine the 'AMD' Adj Close from the stocks dictionary
stocks['AMD'].head()


# ## Create the Equal-Weighted Portfolio
# To create the equal-weighted portfolio, we need to add some additional columns to the DataFrames in the `stocks` dictionary. The three columns that we will build are:
# * Normalized Return = Adjusted Close / Adjusted Close on the `startdate` of the portfolio
# * Allocation = Normalized Return * 0.25 (equal weighting for each of the four stocks)
# * Position Value = Allocation * 10,000 (value of the portfolio)

# In[8]:


# Create 'Normalized Return' column for each stock
for stock_name, stock_data in stocks.items():
    first_adj_close = stock_data.iloc[0] ['Adj Close']
    stock_data['Normalized Return'] = stock_data ['Adj Close']/first_adj_close


# In[9]:


stocks['AAPL'].head()


# In[64]:


# Create allocation for each stock - equally weighted in our initial portfolio
for stock_name, stock_data in stocks.items():
    stock_data['Allocation'] = stock_data['Normalized Return'] * 0.25


# In[65]:


stocks['MSFT'].head()


# In[67]:


# Set the value of the portfolio to $10k
for stock_name, stock_data in stocks.items():
    stock_data['Position Value'] = stock_data ['Allocation'] * 1000


# In[68]:


stocks['ORCL'].head()


# ## Visualize the Portfolio Performance
# To visualize the performance of the portfolio, we can create two line charts that show the return of the portfolio, and the return of the individual stocks, over time. Let's build a new DataFrame that contains just the `position value` for each stock, as well as the total value for the portfolio. We can use this DataFrame to create the two visuals.

# In[69]:


# Create position_values dictionary
position_values = {}

for stock_name, stock_data in stocks.items():

    position_values[stock_name] = stock_data ['Position Value']


# In[70]:


# Convert the position_values dictionary to a DataFrame
position_values = pd.DataFrame(data=position_values)
position_values.head()


# In[71]:


# Add 'Total' column to position values, summing the other columns
position_values['Total'] = position_values.sum(axis=1)
position_values.head()


# In[72]:


plt.figure(figsize=(12, 8))

plt.plot(position_values['Total'])

plt.title('Equal-Weighted Portfolio Performance')
plt.ylabel('Total Value');


# In[73]:


# View the total portfolio
plt.figure(figsize=(12, 8))

plt.plot(position_values.iloc[:,0:4])

plt.title('Equal-Weighted Portfolio Stock Performance')
plt.ylabel('Total Value');


# In[ ]:


# View the four stocks in the portfolio


# ## Calculate Performance Metrics for the Portfolio
# Now that we have created and visualized the equal-weighted portfolio, we can calculate a few metrics to further measure the performance of the portfolio. We will create five performances metrics:
#  * Cumulative Return
#  * Mean Daily Return
#  * Standard Deviation Daily Return
#  * Sharpe Ratio
#  * Annualized Sharpe Ratio

# In[74]:


# Define the end and start value of the portfolio
end_value= position_values['Total'][-1]
start_value= position_values['Total'][0]

# Calculate the cumulative portfolio return as a percentage
cumulative_return = end_value/start_value -1
print(cumulative_return)


# In[75]:


# Create a 'Daily Returns' column
position_values['Daily Returns'] = position_values['Total'].pct_change()
position_values.head()


# In[76]:


# Calculate the mean Daily Return 
mean_daily_return = position_values['Daily Returns'].mean()

print('The mean daily return is:', str(mean_daily_return))


# In[77]:


# Calculate the standard deviation of Daily Return 
std_daily_return = position_values['Daily Returns'].std()

print('The std daily return is:',(std_daily_return)) 


# ### Sharpe Ratio
# Now, let's explore a risk-adjusted return metric called the sharpe ratio. The sharpe ratio helps us to quantify how much return we are getting for a given level of risk. When comparing two different investments, the asset with the higher sharpe ratio provides a higher return for the same amount of risk or the same return for a lower amount of risk. 
# 
# It is calculated by taking the average return of the portfolio, minus a risk free rate (such as government bonds), divided by the standard deviation of the return. In this case, we assume the risk-free rate is close 0 so we won't add it to the formula.

# In[78]:


# Calculate the sharpe ratio
sharpe_ratio = mean_daily_return / std_daily_return
sharpe_ratio


# In[82]:


# Calculate the annualized sharpe ratio
sharpe_ratio_annualized = sharpe_ratio * 252**0.5

sharpe_ratio_annualized


# ## Prepare Scenarios to Optimize Portfolio Weighting
# We need to prepare our data ahead of generating our scenarios to optimize the portfolio weighting. We will:
#  * Create a dictionary containing the adjusted close for each of our stocks: stock_adj_close
#  * Create another dictionary that transforms the adjusted close for each day to a percent change from the previous day

# In[79]:


# Create stock_adj_close dictionary
stock_adj_close = {}
for stock_name, stock_data in stocks.items():
    
    stock_adj_close[stock_name] = stock_data['Adj Close']
    


# In[80]:


stock_adj_close =pd.DataFrame(data=stock_adj_close)
stock_adj_close.head()


# In[81]:


# Create stock_returns DataFrames to see the day over day change in stock value
stock_returns=stock_adj_close.pct_change()
stock_returns.head()


# ## Build & Run 10,000 Portfolio Scenarios
# Now that we've prepared our data, we're almost ready to run our scenarios. First, we need to build the structures required to generate these scenarios and store the output. To do this, we will use the `numpy.zeros()` function. 
# 
# This function creates arrays that are filled with zeros. After we run the scenarios, we replace these zeros with the corresponding output. The reason we create the arrays with zeros first is to give our arrays the correct shape before we replace them with the correct values.
# 
# We will create four different arrays:
#  * weights_array - this array will have 10,000 rows and 4 columns and hold the weighting allocation for each stock
#  * returns_array - this array will contain the portfolio return for each scenario
#  * volatility_array - this array will contain the portfolio volatility for each scenario
#  * sharpe_array - this array will contain the sharpe ratio for each scenario

# In[43]:


# Define the number of scenarios and create a blank array to populate stock weightings for each scenario
scenarios = 10000

weights_array = np.zeros((scenarios, len(stock_returns.columns)))

weights_array


# In[45]:


# Create additional blank arrays for scenario output
returns_array = np.zeros(scenarios)
volatility_array = np.zeros(scenarios)
sharpe_array = np.zeros(scenarios)


# In[48]:


import random
random.seed(3)
np.random.seed(3)

for index in range(scenarios): 
    # Generate four random numbers for each index
    numbers = np.array(np.random.random(4))
    
    # Divide each number by the sum of the numbers to generate the random weight
    weights = numbers / np.sum(numbers)
    
    # Save the weights in weights_array
    weights_array[index,:] = weights
    
    # Calculate the return for each scenario
    returns_array[index] = np.sum(stock_returns.mean()*252*weights)
    
    # Calculate the expected volatility for each scenario
    volatility_array[index] = np.sqrt(np.dot(weights.T,np.dot(stock_returns.cov()*252, weights)))

    # Calculate the Sharpe Ratio for each scenario 
    sharpe_array[index] = returns_array[index] / volatility_array[index]


# In[49]:


print("The first combination:", weights_array[0])


# In[50]:


print("The sharpe ratio of the first portfolio:",sharpe_array[0])


# ## Identify the Optimal Portfolio
# Now that we have the output for all 10,000 scenarios, we can identify the optimal portfolio. The optimal portfolio in this case study is the portfolio that has the highest sharpe ratio.

# In[51]:


# Find the highest sharpe ratio in sharpe_array
sharpe_array.max()


# In[52]:


# Find the index of the optimal portfolio
index_max_sharpe = sharpe_array.argmax()
index_max_sharpe


# In[53]:


# Print the optimal weights for each stock
print(stock_list)
print(weights_array[index_max_sharpe,:])


# ## Visualize the Optimal Portfolio & Portfolio Scenarios
# Let's visualize our portfolio scenarios by using a scatter chart. We can use the volatility and returns arrays on each axis to see the relationship between risk and reward. As a final step, we can visualize where the optimal portfolio appears among all of the scenarios.

# In[54]:


# Visualize volatility vs returns for each scenario
plt.figure(figsize=(12,8))

plt.scatter(volatility_array, returns_array, c=sharpe_array, cmap='viridis')

plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return');


# In[56]:


# Identify the optimal portfolio in the returns and volatility arrays
max_sharpe_return = returns_array[index_max_sharpe]
max_sharpe_volatility = volatility_array[index_max_sharpe]

# Visualize volatility vs returns for each scenario
plt.figure(figsize=(12,8))

plt.scatter(volatility_array, returns_array, c=sharpe_array, cmap='viridis')

plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return');

# Add the optimal portfolio to the visual
plt.scatter(max_sharpe_volatility, max_sharpe_return, c='orange', edgecolors='black');

