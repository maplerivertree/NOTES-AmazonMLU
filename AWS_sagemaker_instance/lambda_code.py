import os
import io
import boto3
import json
import csv
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
def lambda_handler(event, context):
    payload = str(event)
    payload = payload.replace("'", '"')
    sm=boto3.client("runtime.sagemaker")
    response = sm.invoke_endpoint(
	    EndpointName= ENDPOINT_NAME,
	    Body=payload,
	    ContentType="application/json",
	    Accept="application/json")
    a = int(response['Body'].read().decode()[6])
    if(a == 0):
        answer = "product safety issue"
    else:
        answer = "not product safety issue"
    return answer
