
import json

import boto3
import numpy as np
import tensorflow as tf

s3 = boto3.client('s3')

# Loading model
model_path = './00000001/'
loaded_model = tf.saved_model.load(model_path)
infer = loaded_model.signatures['serving_default']

def handler(event, context):
    print('Received event: ' + json.dumps(event, indent=2))

    destination = '/tmp/' + event["file"]
    s3.download_file(event["bucket"], event["prefix"] + event["file"], destination)
    data = np.load(destination)

    predictions = infer(tf.constant(data, dtype=tf.float32))['dense_output']
    predictions = predictions.numpy().squeeze(-1).tolist()
    print('predictions: {}'.format(predictions))

    return {
        'statusCode': 200,
        'body': json.dumps(predictions)
    }