# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 20:58:58 2018

@author: Jake
"""

#
# CS 260 Example
# Logistic Regression Example
#
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

# Load the handwritten digits dataset
digits = datasets.load_digits()

# Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
        test_size=0.25, random_state=0)

# Create logistic regression object
regr = linear_model.LogisticRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
digits_y_pred = regr.predict(X_test)

# Calculate the score
score = regr.score(X_test, y_test)
print(score)

# 0.953333333333

# Print to show there are 1797 images (8x8 images for a dimensionality of 64)
print("Image data shape:", digits.data.shape)

# Print to show there are 1797 labels as well (integers 0-9)
print("Label data shape:", digits.target.shape)

# Print the first 5 images along with their labels
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index+1)
    plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize=20)
plt.show()