import os
from data_processing import DBops
from agent import ResponseAgent

# Setup environment variables
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_FILE_KEY = os.getenv("S3_FILE_KEY")

# Ensure the database is setup before handling any events
db_ops = DBops()
db_ops.setup_database()

def lambda_handler(event, context):
    print("Lambda handler started")
    
    db_ops.process_file_from_s3(S3_BUCKET_NAME, S3_FILE_KEY)

    agent = ResponseAgent()  
    print("Agent initialised!")
    user_query = event.get('currentIntent', {}).get('slots', {}).get('QuerySlot', 'Default query if not provided')
    response_message = agent.answer_question(user_query)

    print("Returning response from Lambda")

    return {
        "dialogAction": {
            "type": "Close",
            "fulfillmentState": "Fulfilled",
            "message": {
                "contentType": "PlainText",
                "content": response_message
            }
        }
    }

