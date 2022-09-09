import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import neighbors
from sklearn import metrics
from sklearn import naive_bayes
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve

# Machine Learning Project
# Author: Boris Le
# Date: 03/04/2022
# Description: Machine learning project that use ML to guess a person's sexual orientation based on certain
#              characteristics. Uses a dataset of 2000 anonymous dating profiles provided by HackerEarth.


# Read in data from data set
df = pd.read_csv("DatingData.csv")

# Restrict dataframe to students only
df = df[df.job == 'student']


# Originally going to do location but all based in california
# Arrays storing features we believe to have greatest correlation to target
workingSet = ['orientation', 'drinks', 'drugs', 'pets', 'smokes', 'new_languages', 'body_profile',
            'education_level', 'interests', 'other_interests']

features = ['drinks', 'drugs', 'pets', 'smokes', 'new_languages', 'body_profile',
            'education_level', 'interests', 'other_interests']
label = 'orientation'

# Change dataframe to only contain the working set
df = df[workingSet]

#Set up dummy variables and refactor feature array to contain new columns
df = pd.get_dummies(df, columns=features)
features = df.columns.drop('orientation')

# Assign target and attributes to x and y
X, y = df[features].values, df[label].values

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)
print('Training cases: %d\nTest cases: %d' % (X_train.shape[0], X_test.shape[0]))

# Logistic Regression Model
reg = 0.01
model2 = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)
predictions2 = model2.predict(X_test)
print("Logistic Regression Model")
print("-------------------------")
print('Predicted Labels: ', predictions2)
print('Actual Labels: ', y_test)
print('Logistic Regression Accuracy ', accuracy_score(y_test, predictions2))
print(classification_report(y_test, predictions2, zero_division=1))
print()

# Naive Bayes model
model3 = naive_bayes.GaussianNB().fit(X_train, y_train)
predictions3 = model3.predict(X_test)
print("Naive Bayes Model")
print("-------------------------")
print('Predicted Labels: ', predictions3)
print('Actual Labels: ', y_test)
print('Naive Bayes Accuracy ', accuracy_score(y_test, predictions3))
print(classification_report(y_test, predictions3, zero_division=1))
print()


#Nearest Neighbour model
model1 = neighbors.KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
predictions1 = model1.predict(X_test)
print("3 Nearest Neighbours Model")
print("------------------------")
print('Predicted Labels: ', predictions1)
print('Actual Labels: ', y_test)
print("3NN accuracy:", metrics.accuracy_score(y_test, predictions1))
print(classification_report(y_test, predictions1, zero_division=1))
print()

# Naive Bayes model
model3 = naive_bayes.GaussianNB().fit(X_train, y_train)
predictions3 = model3.predict(X_test)
print("Naive Bayes Model")
print("-------------------------")
print('Predicted Labels: ', predictions3)
print('Actual Labels: ', y_test)
print('Naive Bayes Accuracy ', accuracy_score(y_test, predictions3))
print(classification_report(y_test, predictions3, zero_division=1))
print()

# From looking at the accuracy score of each model, we determine the 3 nearest neighbour model to be optimal
# We will now look for the optimal hyperparameters of the k nearest neighbour model and also graph out the results of
# our current 3 nearest neighbour model


# Learning Curves to measure performance of 3 nearest neighbours model
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#Create pipeline with neighbour of 3
pipe_lr = make_pipeline(StandardScaler(),
                        neighbors.KNeighborsClassifier(n_neighbors=3))

# Creating learning curve
train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr,
                                                        X=X_train,
                                                        y=y_train,
                                                        train_sizes=np.linspace(
                                                            0.1, 1.0, 10),
                                                        cv=4,
                                                        n_jobs=1)

# Assign mean and standard deviation for train and test
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

#Plot out graph
plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='Training accuracy')
plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='Validation accuracy')
plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.0, 1.03])
plt.show()


# 5 fold cross validation to determine best k value for k neighbours model

# Set range for k neighbors
parameter_range = np.arange(1, 10, 1)

# 5 fold cross validation that calculates accuracy
train_score, test_score = validation_curve(neighbors.KNeighborsClassifier(), X, y,
                                           param_name="n_neighbors",
                                           param_range=parameter_range,
                                           cv=5, scoring="accuracy")

# Mean and sd for training
mean_train_score = np.mean(train_score, axis=1)
std_train_score = np.std(train_score, axis=1)

# Mean and sd for test
mean_test_score = np.mean(test_score, axis=1)
std_test_score = np.std(test_score, axis=1)

# Plot mean accuracy scores
plt.plot(parameter_range, mean_train_score,
         label="Training Score", color='b')
plt.plot(parameter_range, mean_test_score,
         label="Cross Validation Score", color='g')

# Plot graph
plt.title("Validation Curve with KNN Classifier")
plt.xlabel("Number of Neighbours")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.legend(loc='best')
plt.show()

# We determine our best nearest neighbour model to be 9, as from the graph, it is the least underfitted and overfitted
# hyperparameter. We create the model here and print out its corresponding results.

model4 = neighbors.KNeighborsClassifier(n_neighbors=9).fit(X_train, y_train)
predictions4 = model4.predict(X_test)
print("9 Nearest Neighbours Model")
print("------------------------")
print('Predicted Labels: ', predictions4)
print('Actual Labels: ', y_test)
print("9NN accuracy:", metrics.accuracy_score(y_test, predictions4))
print(classification_report(y_test, predictions4, zero_division=1))
print()