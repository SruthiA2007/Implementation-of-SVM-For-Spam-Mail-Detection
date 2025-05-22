# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary libraries.
2.Read the dataset and separate the independent and dependent variables.
3.Split the dataset into training and testing.
4.Do preprocessing if needed, in this case vectorization is needed which is done using CountVectorizer()
5.Train the model using SVC() algorithm and .fit()
6.Predict the model on x_test.
7.Measure its accuracy
 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: SRUTHI A


RegisterNumber: 212224240162 
*/
```
```
    import pandas as pd
    data=pd.read_csv("/content/spam.csv",encoding="Windows-1252")
    data.info()

    x=data['v2'].values
    y=data['v1'].values
    x.shape
    y.shape

    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

    from sklearn.feature_extraction.text import CountVectorizer
    cv=CountVectorizer()
    x_train=cv.fit_transform(x_train)
    x_test=cv.transform(x_test)
    x_train

    from sklearn.svm import SVC
    svc=SVC()
    svc.fit(x_train,y_train)
    y_pred=svc.predict(x_test)
    y_pred

    from sklearn.metrics import accuracy_score
    acc=accuracy_score(y_test,y_pred)
    acc
```


## Output:

![image](https://github.com/user-attachments/assets/3e590391-6159-483d-983e-b6e2ce20b536)

![image](https://github.com/user-attachments/assets/b2d2da06-c02a-40ec-b35c-86a291419ba6)

![image](https://github.com/user-attachments/assets/7bfb96cf-ae72-42b0-b77e-fd166e9439cd)

![image](https://github.com/user-attachments/assets/626bdd92-be10-4161-925e-185ca072c7f9)

![image](https://github.com/user-attachments/assets/87e8873a-0839-4ae5-ab1c-f3d0c2d3667d)








## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
