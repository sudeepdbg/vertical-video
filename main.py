# main.py
"""
FastAPI ingestion service for the vertical-video pipeline.

SECURITY FIX (critical): the original version of this file accepted an
arbitrary client-supplied `config` dict and forwarded it, unmodified, all
the way through SQS into `verticalize.process_video(**config)` /
`process_sports_video(**config)` on the worker. Two of the parameters
reachable that way — `whisper_model` and `yolo_weights` — are ultimately
passed to model loaders (`whisper.load_model()`, `ultralytics.YOLO()`)
that can load a local file path or trigger a network fetch. Since the
worker downloads client-uploaded files to a **predictable path**
(`/tmp/{job_id}_in.mp4`), a client could:
  1. Upload an arbitrary file (disguised as a video) via /upload.
  2. Submit a job with `config.whisper_model` (or `.yolo_weights`) pointing
     at that same predictable path.
  3. Have the worker attempt to load their uploaded file as a model
     checkpoint — a well-known deserialization attack surface.

More generally: forwarding a client-controlled dict as `**kwargs` into any
function that touches the filesystem or network is unsafe regardless of
which specific parameters look dangerous today: adding a new kwarg to
verticalize.py later re-opens the hole silently.

Fix: `JobConfigIn` below is a strict Pydantic allowlist. Every field the
client is allowed to influence is declared explicitly, with a type and
(where relevant) bounds or a closed `Literal` set. Fields that must never
be client-controlled — `whisper_model`, `yolo_weights`, and anything else
not listed here — are simply absent from the model, so FastAPI/Pydantic
rejects any request that tries to set them (`extra="forbid"`) instead of
silently passing them through.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
import boto3, uuid, json, os, re

REGION        = "us-east-1"
INGEST_BUCKET = os.environ["INGEST_BUCKET"]
OUTPUT_BUCKET = os.environ["OUTPUT_BUCKET"]
QUEUE_URL     = os.environ["QUEUE_URL"]
TABLE_NAME    = os.environ["TABLE_NAME"]

# ── Server-side-only constants ────────────────────────────────────────────────
# These are intentionally NOT client-settable (see module docstring). If you
# want to offer a choice of Whisper model size, add a *closed* Literal field
# to JobConfigIn (e.g. whisper_size: Literal["tiny","base","small"]) and map
# it to a fixed, server-controlled path/name here — never accept a raw
# string that flows into a loader.
_SERVER_WHISPER_MODEL = "base"
_SERVER_YOLO_WEIGHTS  = "yolov8n.pt"

_ALLOWED_RESOLUTIONS = (
    "Match source (no upscale)",
    "1080p  (1080x1920 - Full HD)",
    "720p   (720x1280  - HD)",
    "540p   (540x960   - SD)",
    "480p   (480x854   - Low)",
)
_ALLOWED_SUBTITLE_STYLES = ("Bold White (TikTok)", "Yellow (Classic)", "Box (Accessible)")
_ALLOWED_MODES = ("subject", "talking_head", "cinematic", "sports_action")
_ALLOWED_SPORTS = ("auto", "basketball", "football", "soccer", "hockey")
_JOB_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class JobConfigIn(BaseModel):
    """
    Strict allowlist of client-settable rendering options.

    `extra="forbid"` (set in model_config below) means any field NOT listed
    here — including `whisper_model`, `yolo_weights`, or anything a future
    verticalize.py refactor might add — is rejected with a 422 instead of
    silently passing through. This is the core of the fix: the allowlist
    is enforced by construction, not by remembering to exclude things.
    """
    model_config = {"extra": "forbid"}

    resolution_label: Literal[_ALLOWED_RESOLUTIONS] = "720p   (720x1280  - HD)"  # type: ignore[valid-type]
    crf: int = Field(default=23, ge=15, le=35)
    encoder_preset_label: Literal["ultrafast", "fast", "medium", "slow"] = "fast"
    confidence: float = Field(default=0.45, ge=0.05, le=0.99)
    smooth_window: int = Field(default=15, ge=3, le=31)
    adaptive_smoothing: bool = True
    use_optical_flow: bool = True
    rule_of_thirds: bool = True
    scene_cut_threshold: float = Field(default=0.35, ge=0.05, le=0.90)
    talking_head_bias: float = Field(default=0.30, ge=0.0, le=1.0)

    burn_subtitles: bool = False
    whisper_language: Optional[str] = Field(default=None, max_length=8)
    subtitle_style_name: Literal[_ALLOWED_SUBTITLE_STYLES] = "Bold White (TikTok)"  # type: ignore[valid-type]
    subtitle_max_chars: int = Field(default=42, ge=20, le=60)
    subtitle_translate_to: Optional[str] = Field(default=None, max_length=8)

    audio_bitrate_label: Literal["64k", "96k", "128k", "192k"] = "128k"

    sport_type: Literal[_ALLOWED_SPORTS] = "auto"
    use_ball_tracking: bool = True
    use_kalman: bool = True

    panel_mode_override: Literal["auto", "force_on", "force_off"] = "auto"

    @field_validator("whisper_language", "subtitle_translate_to")
    @classmethod
    def _lang_code_shape(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not re.fullmatch(r"[a-zA-Z-]{2,8}", v):
            raise ValueError("must look like a language code, e.g. 'en' or 'zh-CN'")
        return v

    def to_verticalize_kwargs(self) -> dict:
        """
        Expand into the exact kwargs verticalize.process_video /
        process_sports_video expect, injecting the SERVER-CONTROLLED model
        names rather than anything client-supplied.
        """
        d = self.model_dump()
        d["whisper_model"] = _SERVER_WHISPER_MODEL
        d["yolo_weights"]  = _SERVER_YOLO_WEIGHTS
        return d


class CreateJobIn(BaseModel):
    model_config = {"extra": "forbid"}
    job_id: str
    mode: Literal[_ALLOWED_MODES] = "subject"
    preset: Literal[_ALLOWED_RESOLUTIONS] = "720p   (720x1280  - HD)"  # type: ignore[valid-type]
    config: JobConfigIn = Field(default_factory=JobConfigIn)

    @field_validator("job_id")
    @classmethod
    def _job_id_shape(cls, v: str) -> str:
        if not _JOB_ID_RE.match(v):
            raise ValueError("job_id must be a UUID (as returned by /upload)")
        return v


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

s3       = boto3.client("s3",        region_name=REGION)
sqs      = boto3.client("sqs",       region_name=REGION)
dynamodb = boto3.resource("dynamodb", region_name=REGION)
table    = dynamodb.Table(TABLE_NAME)

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_filename(name: str) -> str:
    """Strip path separators / odd characters and cap length before using a
    client-supplied filename as part of an S3 key."""
    name = os.path.basename(name or "upload.mp4")
    name = _SAFE_FILENAME_RE.sub("_", name)
    return name[-128:] or "upload.mp4"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    safe_name = _sanitize_filename(file.filename)
    s3_key = f"uploads/{job_id}/{safe_name}"
    s3.upload_fileobj(file.file, INGEST_BUCKET, s3_key)
    table.put_item(Item={
        "job_id": job_id, "status": "UPLOADED",
        "s3_key": s3_key, "filename": safe_name,
    })
    return {"job_id": job_id, "s3_key": s3_key}


@app.post("/jobs")
def create_job(body: CreateJobIn):
    # FIXED (critical): `body` is now a validated JobConfigIn/CreateJobIn,
    # not a raw dict. Every field that reaches verticalize.py has an
    # explicit type and, for anything that could touch the filesystem or a
    # loader, a closed set of allowed values. whisper_model/yolo_weights
    # are never read from the client at all (see JobConfigIn.to_verticalize_kwargs).
    item = table.get_item(Key={"job_id": body.job_id}).get("Item")
    if not item:
        raise HTTPException(404, "job_id not found — upload first")
    if "s3_key" not in item:
        # FIXED: previously this would raise a raw KeyError -> unhandled 500
        # further down when the worker tried to read it. Fail clearly here.
        raise HTTPException(409, "job_id has no associated upload; re-upload first")

    config = body.config.to_verticalize_kwargs()
    config["tracking_mode"] = body.mode
    config["target_preset_label"] = body.preset

    table.update_item(
        Key={"job_id": body.job_id},
        UpdateExpression="SET #s = :s",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":s": "PENDING"},
    )
    sqs.send_message(
        QueueUrl=QUEUE_URL,
        MessageBody=json.dumps({
            "job_id": body.job_id,
            "s3_key": item["s3_key"],
            "config": config,
        }),
    )
    return {"job_id": body.job_id, "status": "PENDING"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str = PathParam(..., pattern=_JOB_ID_RE.pattern)):
    item = table.get_item(Key={"job_id": job_id}).get("Item")
    if not item:
        raise HTTPException(404, "Not found")
    return item
