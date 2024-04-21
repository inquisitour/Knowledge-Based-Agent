import os
import boto3
import hashlib
import pandas as pd
from io import BytesIO
from data_preprocessing import DBops, get_database_url
from inference_engine import ResponseAgent

# Environment variables for database and API configuration
DATABASE_URL = get_database_url()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_FILE_KEY = os.getenv("S3_FILE_KEY")

def lambda_handler(event, context):
    # Initialize AWS S3 client
    s3_client = boto3.client('s3')

    # Initialize the Response Agent
    agent = ResponseAgent()

    try:
        # Retrieve and check the data file from S3
        obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=S3_FILE_KEY)
        file_content = obj['Body'].read()
        file_hash = DBops.calculate_file_hash(file_content)

        if DBops.check_data_hash(file_hash):
            print("Data is up-to-date.")
        else:
            DBops.update_data_hash(file_hash)
            csv_data = pd.read_csv(BytesIO(file_content))
            print("Data hash and content updated.")
            
    except Exception as e:
        print(f"Error handling S3 object: {e}")
        return {'statusCode': 500, 'body': 'Error handling S3 object'}

    # Process Lex event and generate a response using ResponseAgent
    user_query = event['currentIntent']['slots']['QuerySlot']  
    response_message = agent.answer_question(user_query)  # Ensure this method exists and is properly implemented in ResponseAgent

    # Format the response for Lex
    lex_response = {
        "dialogAction": {
            "type": "Close",
            "fulfillmentState": "Fulfilled",
            "message": {
                "contentType": "PlainText",
                "content": response_message
            }
        }
    }

    return lex_response
