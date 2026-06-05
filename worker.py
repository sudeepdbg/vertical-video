# worker.py
import boto3, json, os, urllib.request
import verticalize

sqs = boto3.client('sqs')
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['TABLE_NAME'])

def update_status(job_id, status, progress):
    table.update_item(Key={'job_id': job_id}, 
        UpdateExpression='SET #s = :s, progress = :p',
        ExpressionAttributeNames={'#s': 'status'},
        ExpressionAttributeValues={':s': status, ':p': progress})

def process(msg):
    body = json.loads(msg['Body'])
    job_id, s3_key = body['job_id'], body['s3_key']
    config = body.get('config', {})
    
    in_path, out_path = f'/tmp/{job_id}_in.mp4', f'/tmp/{job_id}_out.mp4'
    update_status(job_id, 'PROCESSING', 10)
    
    s3.download_file(os.environ['INGEST_BUCKET'], s3_key, in_path)
    
    # The magic bridge to your code!
    def cb(val, msg_txt): 
        update_status(job_id, 'PROCESSING', 10 + int(val * 80))
        
    verticalize.process_video(in_path, out_path, progress_callback=cb, **config)
    
    s3.upload_file(out_path, os.environ['OUTPUT_BUCKET'], f"processed/{job_id}.mp4")
    update_status(job_id, 'COMPLETED', 100)
    os.remove(in_path); os.remove(out_path)

while True:
    resp = sqs.receive_message(QueueUrl=os.environ['QUEUE_URL'], MaxNumberOfMessages=1, WaitTimeSeconds=20)
    for msg in resp.get('Messages', []):
        try:
            process(msg)
            sqs.delete_message(QueueUrl=os.environ['QUEUE_URL'], ReceiptHandle=msg['ReceiptHandle'])
        except Exception as e:
            print(f"Failed: {e}")
            # Update DynamoDB to FAILED here
