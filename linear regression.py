#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics


# In[13]:


data=pd.read_csv('prof.csv')
data


# In[16]:


df=pd.DataFrame(data)


# In[17]:


df.columns


# In[35]:


df=df.drop(['local_name', 'take_again', 'diff_index',
       'tag_professor', 'num_student', 'post_date', 'name_onlines',
       'name_not_onlines', 'student_star', 'student_difficult', 'attence',
       'for_credits', 'would_take_agains', 'grades', 'help_useful',
       'help_not_useful', 'comments', 'word_comment', 'gender', 'race',
       'asian', 'hispanic', 'nh_black', 'nh_white', 'gives_good_feedback',
       'caring', 'respected', 'participation_matters',
       'clear_grading_criteria', 'skip_class', 'amazing_lectures',
       'inspirational', 'tough_grader', 'hilarious', 'get_ready_to_read',
       'lots_of_homework', 'accessible_outside_class', 'lecture_heavy',
       'extra_credit', 'graded_by_few_things', 'group_projects', 'test_heavy',
       'so_many_papers', 'beware_of_pop_quizzes', 'IsCourseOnline'],axis=1).drop_duplicates(subset=['professor_name'])


# In[38]:


df


# In[39]:


import matplotlib.pyplot as plt
df.plot(x="star_rating",y="edugood_rating",style="o")
plt.xlabel("star_rating")
plt.ylabel("edugood_rating")
plt.show()


# In[63]:


data_=data.loc[:,['star_rating','edugood_rating']]


# In[78]:


X=pd.DataFrame(data['star_rating'])
y=pd.DataFrame(data['edugood_rating'])


# In[ ]:





# In[79]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[80]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[81]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)


# In[82]:


print(regressor.intercept_)


# In[83]:


print(regressor.coef_)


# In[86]:


y_pred = regressor.predict(X)


# In[87]:


y_pred


# In[ ]:




