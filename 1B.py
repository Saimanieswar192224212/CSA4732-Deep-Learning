from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
import seaborn as sns
import matplotlib.pyplot as plt
x,y=load_breast_cancer(return_X_y=True)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
tree=DecisionTreeClassifier(random_state=23)
tree.fit(x_train,y_train)
y_pred=tree.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt='g',xticklabels=['malignant','benign'],
            yticklabels=['malignant','benign'])
plt.xlabel('actual',fontsize=13)
plt.ylabel('prediction',fontsize=13)
plt.title('confusion matrix',fontsize=17)
plt.show()
print("accuracy",accuracy_score(y_test,y_pred))
print("precision",precision_score(y_test,y_pred))
print("recall",recall_score(y_test,y_pred))
print("f1-score",f1_score(y_test,y_pred))