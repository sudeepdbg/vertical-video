# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import boto3, uuid, json, os

REGION        = "us-east-1"
INGEST_BUCKET = os.environ["INGEST_BUCKET"]
OUTPUT_BUCKET = os.environ["OUTPUT_BUCKET"]
QUEUE_URL     = os.environ["QUEUE_URL"]
TABLE_NAME    = os.environ["TABLE_NAME"]

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

s3       = boto3.client("s3",        region_name=REGION)
sqs      = boto3.client("sqs",       region_name=REGION)
dynamodb = boto3.resource("dynamodb", region_name=REGION)
table    = dynamodb.Table(TABLE_NAME)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    s3_key = f"uploads/{job_id}/{file.filename}"
    s3.upload_fileobj(file.file, INGEST_BUCKET, s3_key)
    table.put_item(Item={
        "job_id": job_id, "status": "UPLOADED",
        "s3_key": s3_key, "filename": file.filename,
    })
    return {"job_id": job_id, "s3_key": s3_key}

@app.post("/jobs")
def create_job(body: dict):
    job_id = body.get("job_id")
    config = body.get("config", {})   # passes straight to verticalize kwargs
    config["tracking_mode"] = body.get("mode", "subject")
    config["target_preset_label"] = body.get("preset", "720p   (720x1280  - HD)")

    # Get s3_key from DynamoDB
    item = table.get_item(Key={"job_id": job_id}).get("Item")
    if not item:
        raise HTTPException(404, "job_id not found — upload first")

    table.update_item(
        Key={"job_id": job_id},
        UpdateExpression="SET #s = :s",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":s": "PENDING"},
    )
    sqs.send_message(
        QueueUrl=QUEUE_URL,
        MessageBody=json.dumps({
            "job_id": job_id,
            "s3_key": item["s3_key"],
            "config": config,
        }),
    )
    return {"job_id": job_id, "status": "PENDING"}

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    item = table.get_item(Key={"job_id": job_id}).get("Item")
    if not item:
        raise HTTPException(404, "Not found")
    return item
