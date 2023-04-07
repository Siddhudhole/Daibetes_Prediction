#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 


# In[2]:


df = pd.read_csv("C:\\Users\\91932\\Downloads\\diabetes.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# In[7]:


df.describe()


# In[10]:


df['BMI'] = df['BMI'].replace(0,df['BMI'].mean())  


# In[11]:


df['Glucose'] = df['Glucose'].replace(0,df['Glucose'].mean())


# In[12]:


df['BloodPressure'] = df['BloodPressure'].replace(0,df['BloodPressure'].mean()) 


# In[13]:


df['SkinThickness'] = df['SkinThickness'].replace(0,df['SkinThickness'].mean()) 


# In[14]:


df['Insulin'] = df['Insulin'].replace(0,df['Insulin'].mean()) 


# In[15]:


df.describe()


# In[16]:


import matplotlib.pyplot as plt 
import seaborn as sns 


# In[17]:


fig,axs = plt.subplots(figsize=(15,10))
sns.boxplot(data=df,width=0.5,ax=axs,fliersize=3)


# In[18]:


df.head()


# In[19]:


x = df.drop('Outcome',axis=1)


# In[20]:


x.head()


# In[21]:


y = df['Outcome']


# In[22]:


from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0 )


# In[23]:


x_train.shape,x_test.shape


# In[24]:


y_train.shape,y_test.shape


# In[27]:


from sklearn.preprocessing import StandardScaler


# In[28]:


import pickle


# In[38]:


def scaler_standard(x_train,x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    file = open('daibetes_scaler.pkl','wb')
    pickle.dump(scaler,file)
    file.close()
    
    return x_train_scaled,x_test_scaled 


# In[39]:


x_train_scaled,x_test_scaled =scaler_standard(x_train,x_test)


# In[40]:


x_train_scaled.shape,y_train.shape


# In[41]:


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()


# In[42]:


log_reg.fit(x_train_scaled,y_train)


# In[44]:


log_reg.get_params()


# In[45]:


from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')


# In[46]:


parameters = {
    'penalty':['l1','l2','elasticnet'],
    'C':np.logspace(-3,3,7),
    'solver':['newton-cg','lbfgs','liblinear']
}


# In[47]:


logreg = LogisticRegression()

clf = GridSearchCV(logreg,
                  param_grid=parameters,
                  scoring='accuracy',
                  cv=10)

clf.fit(x_train_scaled,y_train)


# In[48]:


clf.best_params_


# In[49]:


clf.best_score_


# In[50]:


y_pred = clf.predict(x_test_scaled)


# In[51]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[53]:


con_mat = confusion_matrix(y_test,y_pred)


# In[55]:


con_mat


# In[54]:


true_positive = con_mat[0][0]
false_positive = con_mat[0][1]
false_negative = con_mat[1][0]
true_negative = con_mat[1][1]


# In[56]:


Accuracy = (true_positive+true_negative)/ (true_positive+false_negative + true_negative)
Accuracy


# In[57]:


percision = true_positive/(true_positive+false_positive)
percision


# In[58]:


Recall = true_positive / (true_positive+false_negative)
Recall


# In[61]:


pickle.dump(clf,open('Daibetes_model.pkl','wb'))


# In[ ]:




