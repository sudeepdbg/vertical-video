# worker.py
"""
SQS worker that pulls video-processing jobs and runs them through
verticalize.py.

SECURITY FIX (critical, defense-in-depth): main.py now validates job
config through a strict Pydantic allowlist before it ever reaches SQS (see
main.py's module docstring for the full writeup of the vulnerability this
closes). This worker used to blindly do `verticalize.process_video(**config)`
on whatever dict arrived in the SQS message body — trusting main.py
completely. That's a fragile trust boundary: if a future producer sends
messages to this same queue directly (a bug, a second API deployed against
the same queue, a compromised main.py), this worker would still forward
attacker-controlled kwargs straight into functions that touch the
filesystem/network (whisper_model / yolo_weights -> model loaders).

`_sanitize_config()` below re-applies the same allowlist independently,
so this file is safe even if it's ever the only thing standing between an
SQS message and verticalize.py.

RELIABILITY FIXES:
- Temp file cleanup now runs in `finally`, not just on the success path —
  previously any exception (bad download, verticalize crash, S3 upload
  failure) left the downloaded input and any partial output in /tmp
  forever, which will eventually fill the disk on a long-running worker.
- A background heartbeat extends the SQS visibility timeout while a job is
  processing, so long jobs (a multi-minute Auto-Clip batch) can't be
  silently redelivered mid-run and processed twice concurrently.
"""
import boto3, json, os, logging, threading, time
import verticalize

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("worker")

REGION        = "us-east-1"
INGEST_BUCKET = os.environ["INGEST_BUCKET"]
OUTPUT_BUCKET = os.environ["OUTPUT_BUCKET"]
QUEUE_URL     = os.environ["QUEUE_URL"]
TABLE_NAME    = os.environ["TABLE_NAME"]

# How often to extend the SQS visibility timeout while a job is in flight,
# and by how much each time. Keep VISIBILITY_EXTENSION_SEC comfortably
# larger than HEARTBEAT_INTERVAL_SEC so a missed beat (e.g. GC pause)
# doesn't let the message become visible again mid-job.
HEARTBEAT_INTERVAL_SEC   = 60
VISIBILITY_EXTENSION_SEC = 180

sqs      = boto3.client("sqs",       region_name=REGION)
s3       = boto3.client("s3",        region_name=REGION)
dynamodb = boto3.resource("dynamodb", region_name=REGION)
table    = dynamodb.Table(TABLE_NAME)

# ── Defense-in-depth config allowlist (mirrors main.py's JobConfigIn) ────────
# Anything NOT in this set is dropped before it reaches verticalize.py,
# regardless of what arrives in the SQS message body. whisper_model and
# yolo_weights are deliberately absent — the worker decides those, not the
# message.
_ALLOWED_CONFIG_KEYS = {
    "resolution_label", "crf", "encoder_preset_label", "confidence",
    "smooth_window", "adaptive_smoothing", "use_optical_flow",
    "rule_of_thirds", "scene_cut_threshold", "talking_head_bias",
    "burn_subtitles", "whisper_language", "subtitle_style_name",
    "subtitle_max_chars", "subtitle_translate_to", "audio_bitrate_label",
    "sport_type", "use_ball_tracking", "use_kalman", "panel_mode_override",
    "target_preset_label",
}
_SERVER_WHISPER_MODEL = "base"
_SERVER_YOLO_WEIGHTS  = "yolov8n.pt"


def _sanitize_config(raw_config: dict) -> dict:
    """Re-apply the allowlist independently of main.py's validation, and
    force the model-loading parameters to fixed, server-controlled values
    no matter what the message contains."""
    if not isinstance(raw_config, dict):
        log.warning("job config was not a dict (%r); ignoring entirely", type(raw_config))
        raw_config = {}
    dropped = set(raw_config.keys()) - _ALLOWED_CONFIG_KEYS
    if dropped:
        log.warning("dropping unexpected config keys from job message: %s", sorted(dropped))
    clean = {k: v for k, v in raw_config.items() if k in _ALLOWED_CONFIG_KEYS}
    clean["whisper_model"] = _SERVER_WHISPER_MODEL
    clean["yolo_weights"]  = _SERVER_YOLO_WEIGHTS
    return clean


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


class _VisibilityHeartbeat:
    """Periodically extends the SQS message's visibility timeout while a
    job is processing, so a slow job doesn't get redelivered and run twice
    concurrently. Runs in a daemon thread; stop() is safe to call multiple
    times and from a finally block."""

    def __init__(self, receipt_handle: str, job_id: str):
        self._receipt_handle = receipt_handle
        self._job_id = job_id
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.wait(HEARTBEAT_INTERVAL_SEC):
            try:
                sqs.change_message_visibility(
                    QueueUrl=QUEUE_URL,
                    ReceiptHandle=self._receipt_handle,
                    VisibilityTimeout=VISIBILITY_EXTENSION_SEC,
                )
                log.debug("job %s: extended SQS visibility timeout", self._job_id)
            except Exception as exc:
                # Don't crash the worker over a heartbeat hiccup — worst
                # case the message becomes visible again and is reprocessed,
                # which is the same risk as not having a heartbeat at all.
                log.warning("job %s: failed to extend visibility timeout: %s",
                           self._job_id, exc)

    def stop(self) -> None:
        self._stop_event.set()


def process(msg):
    body   = json.loads(msg["Body"])
    job_id = body["job_id"]
    s3_key = body["s3_key"]
    config = _sanitize_config(body.get("config", {}))

    in_path  = f"/tmp/{job_id}_in.mp4"
    out_path = f"/tmp/{job_id}_out.mp4"

    heartbeat = _VisibilityHeartbeat(msg["ReceiptHandle"], job_id)
    heartbeat.start()

    # FIXED (reliability): cleanup now lives in `finally` so a download
    # failure, a verticalize crash, or an S3 upload failure can no longer
    # leave input/output files behind in /tmp forever.
    try:
        update_status(job_id, "PROCESSING", 10)
        log.info("Downloading s3://%s/%s", INGEST_BUCKET, s3_key)
        s3.download_file(INGEST_BUCKET, s3_key, in_path)

        def cb(val, msg_txt=""):
            pct = 10 + int(val * 80)
            update_status(job_id, "PROCESSING", pct)
            log.info("  [%d%%] %s", pct, msg_txt)

        mode = config.pop("tracking_mode", None) or "subject"
        if mode == "sports_action":
            meta = verticalize.process_sports_video(in_path, out_path,
                                                     progress_callback=cb, **config)
        elif mode == "cinematic":
            meta = verticalize.process_cinematic_video(in_path, out_path,
                                                        progress_callback=cb, **config)
        else:
            meta = verticalize.process_video(in_path, out_path, tracking_mode=mode,
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
        update_expr = "SET #s = :s, progress = :p, output_url = :u"
        update_vals = {":s": "COMPLETED", ":p": 100, ":u": url}
        # Surface any non-fatal processing warnings (e.g. subtitles were
        # requested but transcription failed) so the frontend/API consumer
        # can show them instead of the job silently looking "fully clean".
        warnings = (meta or {}).get("warnings") or []
        if warnings:
            update_expr += ", warnings = :w"
            update_vals[":w"] = warnings
        table.update_item(
            Key={"job_id": job_id},
            UpdateExpression=update_expr,
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues=update_vals,
        )
        log.info("Job %s COMPLETED%s", job_id,
                f" with {len(warnings)} warning(s)" if warnings else "")
    finally:
        heartbeat.stop()
        for p in (in_path, out_path):
            try: os.remove(p)
            except OSError: pass


def main_loop():
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
                try:
                    update_status(job_id, "FAILED", error=e)
                except Exception:
                    log.exception("Job %s: also failed to record FAILED status", job_id)
                # FIXED (reliability): don't delete the message on failure.
                # Let SQS's own redrive policy (maxReceiveCount -> DLQ,
                # configured on the queue) decide whether to retry or park
                # this in a dead-letter queue for inspection. Deleting here
                # unconditionally — as the original code did — meant a
                # single transient failure (an S3 blip, a momentary OOM)
                # was treated identically to a permanent one: silently
                # dropped, with no retry and no DLQ record.


if __name__ == "__main__":
    main_loop()
