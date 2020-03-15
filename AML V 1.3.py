#!/usr/bin/env python
# coding: utf-8

# In[23]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[24]:


import pandas as pd
import numpy as np
import os
import math
from numpy import exp


# In[25]:


House_data = pd.read_csv('energydata_complete.csv', sep=',')


# In[271]:


House_data.columns


# In[275]:





# In[296]:


# Analyzing data, Taking a small dataset, Dropping date, leaving Appliance enery usage for Y column
# We can use the same x for regression and classification. 
House_data.head(3)
x = House_data.iloc[:,2:]

y = House_data.iloc[:,1]
y_log = (y>50)*1
x.shape
y.shape
y_log.shape


# In[ ]:





# In[277]:


# Split train and test
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.4, random_state=100)
x_train.shape
x_test.shape

# Splitting for classification as well. 
x_train_log,x_test_log,y_train_log,y_test_log = train_test_split(x,y_log,test_size = 0.4, random_state=100)
x_train_log.shape
x_test_log.shape


# In[278]:


# Creating Theta matrix, adding a column for the bias terms

np.random.seed(0)
thetas_matrix = np.random.rand(x_train.shape[1]+1,1)
thetas_matrix.shape


# In[279]:


thetas_matrix


# In[280]:


# Changing DS from Pandas series to numpy arrays
x_train_numpy = np.array(x_train)
x_test_numpy = np.array(x_test)
y_train_numpy = np.array(y_train)
y_test_numpy = np.array(y_test)

x_train_log = np.array(x_train_log)
x_test_log = np.array(x_test_log)
y_train_log = np.array(y_train_log)
y_test_log = np.array(y_test_log)

x_train_numpy.shape
y_train_numpy.shape
x_test_numpy.shape
y_test_numpy.shape
y_test_log.shape
y_train_log.shape

# Here we observe that we need to reshape the y arrays.

# Reshaping Y variables to be M x 1
y_train_numpy = y_train_numpy.reshape(x_train_numpy.shape[0], 1)
y_test_numpy  = y_test_numpy.reshape(x_test_numpy.shape[0],1)

y_train_log  = y_train_log.reshape(x_train_log.shape[0], 1)
y_test_log = y_test_log.reshape(x_test_log.shape[0], 1)


# In[281]:


# Normalization/Standardization 
# Normalizing the features using standard deviation
# 
x_train_numpy = (x_train_numpy - np.mean(x_train_numpy))/np.std(x_train_numpy)
x_test_numpy = (x_test_numpy - np.mean(x_test_numpy))/np.std(x_test_numpy)


x_train_log = (x_train_log - np.mean(x_train_log))/np.std(x_train_log)
x_test_log = (x_test_log - np.mean(x_test_log))/np.std(x_test_log)


# In[282]:


# Appending ones to training data & Test data for the bias term
x_train_numpy = np.insert(x_train_numpy,0,1,axis=1)
x_train_numpy.shape

x_test_numpy = np.insert(x_test_numpy,0,1,axis=1)
x_test_numpy.shape

# Appending ones to training data & Test data for the bias term for the classification x matrix 
x_train_log = np.insert(x_train_log,0,1,axis=1)
x_train_log.shape
x_test_log = np.insert(x_test_log,0,1,axis=1)
x_train_log.shape


# In[10]:


# Here I try to combine the two regression functions
# Make a general runner/wrapper function that takes the data set input 
# Activation = sigmoid or 1 we will use for logistic and the other can be used for regression. 


# In[283]:


# This 
def wrapper_function(activation,alpha,maximum_iterations,data_matrix,y_matrix,theta_matrix):
        
        optimum_thetas,Cost_history,gradient_diff_sum_array= gradient_descent(activation,alpha,data_matrix,theta_matrix,y_matrix,maximum_iterations)
        return optimum_thetas,Cost_history,gradient_diff_sum_array


# In[284]:


def gradient_descent(activation,alpha,data_matrix,theta_matrix,y_matrix,maximum_iterations):
    
    
    
    cost_history_array = np.zeros(maximum_iterations) # for each iteration, keep a cost for plotting
    optimum_thetas = theta_matrix
    updated_thetas = theta_matrix
    gradient_diff_sum_array = np.zeros(maximum_iterations)
    convergence_iteration_number = 0 
    #convergence = False
    for i in range(maximum_iterations):

        print('Epoch : ',i)
        old_cost = cost_function(activation,data_matrix,updated_thetas,y_matrix)
        print('Cost :',old_cost)
        cost_history_array[i] = old_cost
        
       
        gradient = gradient_calculator(activation,data_matrix,updated_thetas,y_matrix,alpha)
        
        # Checking, for debuggin only 
        #gradient_difference = check_gradient(activation,gradient,updated_thetas,data_matrix,y_matrix)
        #gradient_diff_sum_array[i] = np.sum(gradient_difference)
        old_thetas = updated_thetas
        updated_thetas = gradient_updater(gradient,updated_thetas,alpha)
        
        # Calling convergence check function 
        convergence = convergence_check(activation,updated_thetas,old_thetas,data_matrix,y_matrix)
        if(convergence):
            #print('convergence has occured, killing the loop')
            optimum_thetas = old_thetas
            convergence_iteration_number = i
            break

        
        if(i == maximum_iterations-1):
            #print('Try more Epochs for saturation, Did not converge')
            optimum_thetas = updated_thetas

        # Calling gradient checker function
        #gradient_approximation,grad_difference = gradient_checker(gradient,updated_thetas,data_matrix,y_matrix)

            
        #print('-------------------------------------------------------------------------------------------------------')
    return optimum_thetas,cost_history_array,gradient_diff_sum_array
        
        
    


# In[285]:


def predicted_value(data_matrix,thetas):
    return np.dot(data_matrix,thetas)


# In[286]:


def sigmoid_function(z):
    return 1 / (1 + exp(-z))


# In[287]:


def H_function(data_matrix,thetas):
    H = sigmoid_function(predicted_value(data_matrix,thetas)) # here the sigmoid is fed z value
    return H


# In[288]:


def gradient_calculator(activation,data_matrix,thetas,y_matrix,alpha):
    
    Xtheta = predicted_value(data_matrix,thetas) 
    data_matrix_T = np.transpose(data_matrix)
    m = data_matrix.shape[0]
    
    if(activation == 0):
        gradient = (np.dot(data_matrix_T,(Xtheta-y_matrix)))
        return gradient/m
    
    elif(activation == 1):
        h = H_function(data_matrix,thetas)
        gradient = np.dot(data_matrix_T,(h- y_matrix))
        return gradient/m


# In[289]:


def gradient_updater(gradient,updated_thetas,alpha):
    return updated_thetas-(gradient*alpha)


# In[290]:


def convergence_check(activation,updated_thetas,old_thetas,data_matrix,y_matrix):
    if(activation==0):
        old_cost = cost_function(activation,data_matrix,old_thetas,y_matrix)
        updated_cost = cost_function(activation,data_matrix,updated_thetas,y_matrix)
        if(old_cost-updated_cost< 0.001):
            print('Linear Regression Cost has converged,returning thetas in the 2nd last iteration')
            return True
    if(activation==1):
        old_cost = cost_function(activation,data_matrix,old_thetas,y_matrix)
        updated_cost = cost_function(activation,data_matrix,updated_thetas,y_matrix)
        
        if(old_cost-updated_cost< 0.000001):
            print('logistic regression Cost has converged,returning thetas in the 2nd last iteration')
            return True
        


# In[291]:


def cost_function(activation,data_matrix,thetas,y_matrix):
    m = data_matrix.shape[0]
    if(activation ==0):
        predicted = predicted_value(data_matrix,thetas)
        cost = np.dot(np.transpose(predicted - y_matrix),(predicted - y_matrix))
        cost = cost/(2*m)
        return cost
    elif(activation ==1):
        h = H_function(data_matrix,thetas)
        log_h = np.log(h)
        log_1_h = np.log(1-h)
        one_min_y_T = np.transpose((1-y_matrix))
        y_T = np.transpose(y_matrix)
        cost = np.dot((-1*y_T),log_h) - np.dot(one_min_y_T,log_1_h)
        return cost/m
        


# In[292]:


def check_gradient(activation,gradients,thetas,data_matrix,y_values):
    # initializing empty array
    print('check_gradient called')
    Estimated_gradient = np.zeros((thetas.shape[0], 1))
    gradient_difference = np.zeros((thetas.shape[0], 1))
    
    # PSEUDO CODE
    # Thetas are a vector i.e of order n x 1
    # for every theta we have a gradient 
    # gradients are of order n x 1 as well i.e. for each theta
    # We can estimate gradient for each theta that is why we have a loop
    # at each i we are basically estimating ith theta, how ?
    
    # so epsilon vector is n x 1 as well. 
    # when its ith iteration we add 1 to the ith location 
    # we than multiply that epsilon vector with a constant called epsilon constant
    # so at ith iteration we will only have that index having epsilon constant. 
    # so lets say at 5th iteration, we have epsilon vector having epsilon constant at 5th iteration. 
    # we add that to our theta vector so that  only 5th iteration theta changes. 
    # We than compute the estimated derivative 
    # than we can compare it with the gradients computed by gradient descent 
    
    for i in range(thetas.shape[0]):
        epsilon_constant = 0.0001
        epsilon_vector = np.zeros((thetas.shape[0], 1))
        epsilon_vector[i,0] = 1
        theta_iPlus = thetas + np.dot(epsilon_vector,epsilon_constant)
        theta_iMinus = thetas - np.dot(epsilon_vector,epsilon_constant)
        
        Estimated_gradient = cost_function(activation,data_matrix,theta_iPlus,y_values)- cost_function(activation,data_matrix,theta_iMinus,y_values)
        Estimated_gradient = Estimated_gradient/(2*epsilon_constant)
        
        # so our Estimated_gradient is basically a gradient estimate for the ith theta
        #print('Estimated_gradient for  iteration ',i,Estimated_gradient)
        #print('Actual gradient' ,gradients[i])
        
        # So idea is to have the difference for the ith theta estimate and actual value
        gradient_difference[i] = Estimated_gradient - gradients[i,0]
   
        
    return gradient_difference
        
        
        
        
        
    


# In[21]:


# Cost for linear regression is working 
# cost for logistic regression is working as well
# Simoid is working 
# predicted_value is working 


# In[264]:


# Use activation = 1 for sigmoid
# Use activation = 0 for linear regression
# 
activation = 1
alpha = 0.3
maximum_iterations = 15251
data_matrix = x_train_numpy
y_matrix = y_train_log
theta_matrix = thetas_matrix


# In[265]:


optimum_thetas,Cost_history,gradient_diff_sum_array= wrapper_function(activation,alpha,maximum_iterations,x_train_numpy,y_matrix,thetas_matrix)


# In[ ]:



cost_function(1,x_test_numpy,optimum_thetas,y_test_log)
optimum_thetas


# In[192]:


from matplotlib import pyplot as plt

plt.plot(Cost_history)
plt.show()


# In[85]:


# So for each iteration, we have gradients calculated for each theta
# We send those thetas to gradient checker which has estimated gradients
# Gradient checker sends the difference for each gradient back to gradient descent function
# Gradient descent saves the difference of estimated thetas by checker and the calculated thetas by the calculator 
# it saves the difference for each iteration in array
# The check is that this difference sum should be very small if our calculation is correct and estimation is close to it
# hence the difference is less. 
gradient_diff_sum_array


# In[294]:


activation = 0
alpha = 0.01
maximum_iterations = 100000
data_matrix = x_train_numpy
y_matrix = y_train_numpy
theta_matrix = thetas_matrix
optimum_thetas,Cost_history,gradient_diff_sum_array= wrapper_function(activation,alpha,maximum_iterations,data_matrix,y_matrix,thetas_matrix)


# In[124]:


optimum_thetas


# In[295]:


optimum_thetas.shape
cost_function(0,data_matrix,optimum_thetas,y_matrix)
cost_function(0,x_test_numpy,optimum_thetas,y_test_numpy)


# In[87]:


Cost_history


# In[270]:



from matplotlib import pyplot as plt

plt.plot(Cost_history)
plt.show()


# In[ ]:


# Test case with non-zero thetas
x_test = np.array([2,1,3,7,1,9,1,8,1,3,7,4])
x_test=x_test.reshape(4,3)

y_t = np.array([2,5,5,6])
y_t=y_t.reshape(4,1)

theta_test = np.array([0.1,-0.2,0.3])
theta_test=theta_test.reshape(3,1)


alpha = 0.01
m =  x_test.shape[0]
maximum_iterations = 10
activation = 0

#optimum_thetas,Cost_history= wrapper_function(activation,alpha,maximum_iterations,data_matrix,y_matrix,thetas_matrix)

gradient_test_thetas,Cost_history= wrapper_function(activation,alpha,maximum_iterations,x_test,y_t,theta_test)
optimum_thetas,Cost_history


# In[ ]:




