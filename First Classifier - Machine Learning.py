
# coding: utf-8

# In[36]:


from scipy.spatial import distance


# In[37]:


def euc(a,b):
    return distance.euclidean(a,b)


# In[29]:


class NewKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train


# In[38]:


def predict(self, X_train):
    predictions = []
    for row in X_test:
        label = self.closest(row)
        prediction.append(label)
    return predictions


# In[39]:


def closest(self, row):
    best_dist = euc(row, self.X_train[0])
    best_index = 0
    for i in range(1, len(self.X_train)):
        dist = euc(row, self.X_train[i])
        if dist < best_dist:
            best_dist = dist
            best_index = i
    return self.y_train[best_index]


# In[1]:


from sklearn import datasets


# In[3]:


iris = datasets.load_iris()


# In[4]:


X = iris.data


# In[5]:


y = iris.target


# In[6]:


from sklearn.cross_validation import train_test_split


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)


# In[9]:


from sklearn.neighbors import KNeighborsClassifier


# In[27]:


my_classifier = NewKNN()


# In[11]:


my_classifier.fit(X_train, y_train)


# In[12]:


predictions = my_classifier.predict(X_test)


# In[14]:


from sklearn.metrics import accuracy_score


# In[40]:


print(accuracy_score(y_test, predictions))

