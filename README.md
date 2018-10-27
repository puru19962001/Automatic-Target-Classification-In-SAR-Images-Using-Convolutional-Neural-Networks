# Automatic-Target-Classification-In-SAR-Images-Using-Convolutional-Neural-Networks

CNNs -> Convolutional Neural Networks
SAR -> Synthetic Aperture Radar. 

This project is based on predicting the accuracy of the testing data set over the training data set using the MSTAR(Moving and Stationary Target Acquisition and Recognition) database and plotting the graph of the values generated for various targets in Results.csv file.

You need to initially install tensorflow(any version) to run these programs, else the programs will not run.
And also pip install other packages required by the programs. 

First you need to download the MSTAR Database.
You have to sort the database according to the target names in their respective folders.
The target images present in the testing data set must be absent in the training data set.
The testing set must contain less number of target images compared to the training set, this is to achieve better accuracy during the testing process.

Copy the path of the MSTAR Database and paste it in the programs in the respective places.

First run TrainingData.py
Next run Prediction.py

A Results.csv file gets generated.

Plot the bar graph to visualize the accuracy in predicting the targets.
