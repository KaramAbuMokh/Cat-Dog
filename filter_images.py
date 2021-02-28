import shutil
import boto3
import cv2
from botocore.exceptions import NoCredentialsError
import os
from boto.s3.connection import S3Connection
import numpy as np
from PIL import Image

bucket = 'stude'
ACCESS_KEY = 'AKIAZMHSVRRZ43WAGPZB'
SECRET_KEY = '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'
client = boto3.client('rekognition')
s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                  aws_secret_access_key=SECRET_KEY)


def upload_to_aws(local_file, bucket, s3_file):


    try:
        s3.upload_file(local_file, bucket, s3_file)
        print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False


def data():
    # upload the images
    for photo in os.listdir('images'):
        if photo[-3:] == 'png' or photo[-3:] == 'jpg':
            upload_to_aws('images/' + photo, bucket, 'karam/images/' + photo)

    conn = S3Connection(ACCESS_KEY, SECRET_KEY)
    bucket_ls = conn.get_bucket(bucket)
    images_list = []

    
    # get the names with path of the images in the bucket
    for key in bucket_ls.list():
        photo = key.name
        if photo[:13] == 'karam/images/' and (photo[-3:] == 'png' or photo[-3:] == 'jpg'):
            images_list.append(photo[13:])


    # delete the images that isnt in s3
    for photo in os.listdir('images'):
        if photo not in images_list:
            os.remove('images/' + photo)


    for photo in os.listdir('images'):
        try:
            image = Image.open('images/' + photo)
            new_image = image.resize((224, 224))
            new_image.save('images/' + photo)
        except:
            os.remove('images/' + photo)


    # get the high Confidence images names
    names = []
    labels = []
    for photo in images_list:
        print(photo)
        response = client.detect_labels(Image={"S3Object": {"Bucket": bucket, "Name": 'karam/images/' + photo}})
        delete=True
        for obj in response['Labels']:
            if obj['Name'] == 'Cat' and obj['Confidence'] >= 60:
                names.append(photo)
                labels.append([1.0, 0.0])
                delete=False
                break
            if obj['Name'] == 'Dog' and obj['Confidence'] >= 60:
                names.append(photo)
                labels.append([0.0, 1.0])
                delete = False
                break
        if(delete):
            os.remove('images/' + photo)

    return names, labels
