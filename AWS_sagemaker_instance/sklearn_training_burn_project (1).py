from __future__ import print_function

import argparse
import os
import io
from io import BytesIO
import json
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import boto3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    
    # We only have one data file for this example
    train_data = np.load(input_files[0])
    
    print("train_data type", type(train_data))
    print("train_data shape", train_data.shape)
    
    # labels are in the last column
    train_y = train_data[:, 1201]
    train_X = train_data[:, 0:1201]
    
    print("train_y shape:", train_y.shape)
    print("train_x shape:", train_X.shape)
    
    # We will use logistic regression
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(train_X, train_y)

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))

def input_fn(input_data, request_content_type):
    if request_content_type == "application/json":
        
        # Read the data and init s3 client
        data = json.loads(input_data)
        data_records = data["data"]
        sm = boto3.client('s3')
        
        # Let's read TF-IDF vectorizers from the S3 bucket
        # First text vectorizer
        response = sm.get_object(Bucket='mlu-data-example', Key='tfidf_text_vectorizer.pickle')
        model_str = response['Body'].read()
        tfidf_text_vectorizer = pickle.loads(model_str)   
        
        # Then title vectorizer
        response = sm.get_object(Bucket='mlu-data-example', Key='tfidf_title_vectorizer.pickle')
        model_str = response['Body'].read()
        tfidf_title_vectorizer = pickle.loads(model_str)  
        
        # Reading only one record for inference
        data = data_records[0]
        normalized_star = data["star_rating"]/5.0
        text_arr = tfidf_text_vectorizer.transform([str(data["text"]).lower()]).toarray()
        title_arr = tfidf_title_vectorizer.transform([str(data["title"]).lower()]).toarray()
        
        return np.column_stack([normalized_star, text_arr, title_arr])
    
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.
        pass
    
def predict_fn(input_data, model):
    prediction = model.predict(input_data[:, :1201])
    return np.array([prediction])

def output_fn(prediction_result, accept):
    if accept == 'application/json':
        res_df = pd.DataFrame(prediction_result)
        return res_df.to_json(orient='records')
    else:
        raise ValueError('Accept header must be application/json')
    
def model_fn(model_dir):
    """Deserialized and return fitted model
    
    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf
