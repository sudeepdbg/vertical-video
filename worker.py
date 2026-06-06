# worker.py
import boto3, json, os, logging
import verticalize

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("worker")

REGION        = "us-east-1"
INGEST_BUCKET = os.environ["INGEST_BUCKET"]
OUTPUT_BUCKET = os.environ["OUTPUT_BUCKET"]
QUEUE_URL     = os.environ["QUEUE_URL"]
TABLE_NAME    = os.environ["TABLE_NAME"]

sqs      = boto3.client("sqs",       region_name=REGION)
s3       = boto3.client("s3",        region_name=REGION)
dynamodb = boto3.resource("dynamodb", region_name=REGION)
table    = dynamodb.Table(TABLE_NAME)

def update_status(job_id, status, progress=0, error=None):
    expr = "SET #s = :s, progress = :p"
    vals = {":s": status, ":p": progress}
    if error:
        expr += ", error_msg = :e"
        vals[":e"] = str(error)[:500]   # DynamoDB has size limits
    table.update_item(
        Key={"job_id": job_id},
        UpdateExpression=expr,
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues=vals,
    )

def process(msg):
    body   = json.loads(msg["Body"])
    job_id = body["job_id"]
    s3_key = body["s3_key"]
    config = body.get("config", {})

    in_path  = f"/tmp/{job_id}_in.mp4"
    out_path = f"/tmp/{job_id}_out.mp4"

    update_status(job_id, "PROCESSING", 10)
    log.info("Downloading s3://%s/%s", INGEST_BUCKET, s3_key)
    s3.download_file(INGEST_BUCKET, s3_key, in_path)

    def cb(val, msg_txt=""):
        pct = 10 + int(val * 80)
        update_status(job_id, "PROCESSING", pct)
        log.info("  [%d%%] %s", pct, msg_txt)

    mode = config.pop("tracking_mode", "subject")
    if mode == "sports_action":
        verticalize.process_sports_video(in_path, out_path,
                                         progress_callback=cb, **config)
    else:
        verticalize.process_video(in_path, out_path, tracking_mode=mode,
                                  progress_callback=cb, **config)

    out_key = f"processed/{job_id}.mp4"
    log.info("Uploading to s3://%s/%s", OUTPUT_BUCKET, out_key)
    s3.upload_file(out_path, OUTPUT_BUCKET, out_key)

    # Generate presigned URL so frontend can download (valid 24h)
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": OUTPUT_BUCKET, "Key": out_key},
        ExpiresIn=86400,
    )
    table.update_item(
        Key={"job_id": job_id},
        UpdateExpression="SET #s = :s, progress = :p, output_url = :u",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":s": "COMPLETED", ":p": 100, ":u": url},
    )
    log.info("Job %s COMPLETED", job_id)

    # Cleanup temp files
    for p in [in_path, out_path]:
        try: os.remove(p)
        except OSError: pass

while True:
    resp = sqs.receive_message(
        QueueUrl=QUEUE_URL,
        MaxNumberOfMessages=1,
        WaitTimeSeconds=20,
    )
    for msg in resp.get("Messages", []):
        body_preview = json.loads(msg["Body"])
        job_id = body_preview.get("job_id", "unknown")
        try:
            process(msg)
            sqs.delete_message(
                QueueUrl=QUEUE_URL,
                ReceiptHandle=msg["ReceiptHandle"],
            )
        except Exception as e:
            log.error("Job %s failed: %s", job_id, e)
            update_status(job_id, "FAILED", error=e)
            # Still delete the message so it doesn't loop forever
            # Remove this line if you want SQS to retry after visibility timeout
            sqs.delete_message(
                QueueUrl=QUEUE_URL,
                ReceiptHandle=msg["ReceiptHandle"],
            )
