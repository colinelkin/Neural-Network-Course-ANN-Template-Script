"""STEP 1: Preliminary language-specific commands"""
import os
import numpy as np
import pandas as pd
from sklearn import neural_network, model_selection, metrics
from sklearn.preprocessing import MinMaxScaler

"""STEP 2: Load the data"""
# for regression
input_data_reg = np.array(pd.read_csv("Datasets" + os.sep + "input_data_regression_linnerud.csv", header = None)) # replace file names with as needed
output_data_reg = np.array(pd.read_csv("Datasets" + os.sep + "output_data_regression_linnerud.csv", header = None))

# for classification
input_data_class = np.array(pd.read_csv("Datasets" + os.sep + "input_data_classification_digits.csv", header = None)) 
output_data_class = np.array(pd.read_csv("Datasets" + os.sep + "output_data_classification_digits.csv", header = None)) 

"""STEP 3: Determine the ANN hyperparameters"""
# here, we create an entire ANN object with the hyperparameters as inputs
is_class = False # use True if using classification and False if using regression

if is_class: # if classification
    ann = neural_network.MLPClassifier(solver="lbfgs", alpha=.01, hidden_layer_sizes=(20, 10), activation="relu")
else: # if regression
    ann = neural_network.MLPRegressor(solver="sgd", alpha=.01, hidden_layer_sizes=(20,10), activation="logistic")

"""STEP 4: Scale the data (input only under classification; input and output under regression)"""
if is_class: # select the proper data based on whether using classification or regression
    input_data = input_data_class
    output_data = output_data_class
else:
    input_data = input_data_reg
    output_data = output_data_reg

if output_data.ndim == 1:
    output_data = output_data.ravel()

input_scaler = MinMaxScaler(feature_range=(-1, 1)) # scale input data from -1 to 1
input_data = input_scaler.fit_transform(input_data)

if not is_class: # if regression
    output_scaler = MinMaxScaler(feature_range=(0, 1)) # scale output data from 0 to 1
    output_data = output_scaler.fit_transform(output_data)

"""STEP 5: Shuffle the samples and split into train and test"""
[train_in, test_in, train_out, test_out] = model_selection.train_test_split(input_data, output_data, test_size=.2)

"""STEP 6: Train the ANN"""
ann.fit(train_in, train_out) # Easy, isn't it?!

"""STEP 7: Predict training outputs"""
pred_train_out = ann.predict(train_in)

"""STEP 8: Get the training score"""
if is_class: # if classification
    eval_method = "F1 Score"
    # can replace f1_score with different classification score if desired
    train_score = metrics.f1_score(train_out, pred_train_out, average = "macro") 
else: # if regression
    eval_method = "Mean Squared Error"
    train_score = metrics.mean_squared_error(train_out, pred_train_out)

"""STEP 9: Predict testing outputs"""
pred_test_out = ann.predict(test_in)
    
"""STEP 10: Get the testing score"""
if is_class: # if classification
    test_score = metrics.f1_score(test_out, pred_test_out, average = "macro") 
else: # if regression
    test_score = metrics.mean_squared_error(test_out, pred_test_out)
    
"""STEP 11: Save evaluation results and outputs to a file"""
# training and testing results
results = np.array([["Training " + eval_method + " (%): ", 100 * train_score], ["Testing " + eval_method + " (%): ", 100 * test_score]])
results_file = pd.DataFrame(results)
# predicted values versus actual values on training data
train_compare = pd.DataFrame((np.transpose((np.vstack((pred_train_out,np.transpose(train_out)))))))
# predicted values versus actual values on testing data
test_compare = pd.DataFrame((np.transpose((np.vstack((pred_test_out,np.transpose(test_out)))))))

# filepath to "Saved Files" folder
savedir = "Saved Files" + os.sep
# export evaluation results
results_file.to_csv(savedir + eval_method + ".csv", index = False, header = False)
# export training outputs
train_compare.to_csv(savedir + "Training Outputs.csv", index = False, header = ["Predicted", "Actual"])
# export test outputs
test_compare.to_csv(savedir + "Test Outputs.csv", index = False, header = ["Predicted", "Actual"])

"""STEP 12: Display results to the console"""
for elt in results: print(*elt, sep='\n')