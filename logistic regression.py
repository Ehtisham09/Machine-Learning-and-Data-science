import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Social_Network_Ads.csv')
x =dataset.iloc[:,[2,3]].values
y =dataset.iloc[:,4].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

from sklearn.linear_model import LogisticRegression
lin =LogisticRegression(random_state=0)

lin.fit(x_train,y_train)


y_pred = lin.predict(x_test)
from sklearn.metrics import confusion_matrix
cv = confusion_matrix(y_test,y_pred)
print(cv)
from matplotlib.colors import  ListedColormap
x_set,y_set =x_test,y_test
x1,x2 =np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop =x_set[:,0].max()+1,step=0.01),
                   np.arange(start=x_set[:,1].min()-1,stop =x_set[:,1].max()+1,step=0.01))

plt.contourf(x1,x2,lin.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape)
             ,alpha = 0.75,cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i, j in enumerate(np.unique(y_test)):
    plt.scatter(x_set[y_set ==j,0],x_set[y_set ==j,1],
                c = ListedColormap(('red','green'))(i),label = j)

plt.title('Logistic Regression(Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


