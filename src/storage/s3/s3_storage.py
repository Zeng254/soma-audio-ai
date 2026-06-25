import os
import re
from pathlib import Path
from typing import Optional, Any, Dict, List, TypedDict, Iterable
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig
import logging
logger = logging.getLogger(__name__)

# Allowed filename character set（Constraints for user input）
# Note：Does not include / To prevent path traversal
FILE_NAME_ALLOWED_RE = re.compile(r"^[A-Za-z0-9._\-]+$")


class ListFilesResult(TypedDict):
    # list_files Return structure type
    keys: List[str]
    is_truncated: bool
    next_continuation_token: Optional[str]

class S3SyncStorage:
    """S3Compatible storage implementation"""

    def __init__(self, *, endpoint_url: Optional[str] = None, access_key: str, secret_key: str, bucket_name: str, region: str = "cn-beijing"):
        self.endpoint_url = os.environ.get("COZE_BUCKET_ENDPOINT_URL") or endpoint_url or ''
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket_name = bucket_name
        self.region = region
        self._client = None

    def _get_client(self):
        if self._client is None:
            endpoint = self.endpoint_url
            if endpoint is None or endpoint == "":
                try:
                    from coze_workload_identity import Client as CozeEnvClient
                    coze_env_client = CozeEnvClient()
                    env_vars = coze_env_client.get_project_env_vars()
                    coze_env_client.close()
                    for env_var in env_vars:
                        if env_var.key == "COZE_BUCKET_ENDPOINT_URL":
                            endpoint = env_var.value.replace("'", "'\\''")
                            self.endpoint_url = endpoint
                            break
                except Exception as e:
                    logger.error(f"Error loading COZE_BUCKET_ENDPOINT_URL: {e}")
                    # Keep downward validation logic，Avoid interruption here
            if endpoint is None or endpoint == "":
                logger.error("Storage endpoint not configured: please set endpoint_url")
                raise ValueError("Storage endpoint not configured: please set endpoint_url")

            client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region,
            )

            # Register before-call hook, inject x-storage-token header before sending
            def _inject_header(**kwargs):
                try:
                    from coze_workload_identity import Client as CozeClient
                    coze_client = CozeClient()
                    try:
                        token = coze_client.get_access_token()
                    except Exception as e:
                        logger.error("Error loading COZE_WORKLOAD_IDENTITY_TOKEN: %s", e)
                        token = None
                        raise e
                    finally:
                        coze_client.close()
                    params = kwargs.get("params", {})
                    headers = params.setdefault("headers", {})
                    headers["x-storage-token"] = token
                except Exception as e:
                    logger.error("Error loading COZE_WORKLOAD_IDENTITY_TOKEN: %s", e)
                    pass
            client.meta.events.register("before-call.s3", _inject_header)
            self._client = client
        return self._client

    def _generate_object_key(self, *, original_name: str) -> str:
        suffix = Path(original_name).suffix.lower()
        stem = Path(original_name).stem
        uniq = uuid4().hex[:8]
        return f"{stem}_{uniq}{suffix}"

    def _extract_logid(self, e: Exception) -> Optional[str]:
        """Extract x-tt-logid from ClientError"""
        if isinstance(e, ClientError):
            headers = (e.response or {}).get("ResponseMetadata", {}).get("HTTPHeaders", {})
            return headers.get("x-tt-logid")
        return None

    def _error_msg(self, msg: str, e: Exception) -> str:
        """Build error message with logid"""
        logid = self._extract_logid(e)
        if logid:
            return f"{msg}: {e} (x-tt-logid: {logid})"
        return f"{msg}: {e}"

    def _resolve_bucket(self, bucket: Optional[str]) -> str:
        """Unified bucket source parsing, ensures valid bucket name."""
        target_bucket = bucket or os.environ.get("COZE_BUCKET_NAME") or self.bucket_name
        if not target_bucket:
            raise ValueError("Bucket not configured: please provide bucket or set COZE_BUCKET_NAME, or provide bucket_name when instantiating")
        return target_bucket

    def _validate_file_name(self, name: str) -> None:
        """Validate S3 object naming: length<=1024; allows [A-Za-z0-9._-/]; does not start/end with / and does not contain //."""
        msg = (
            "file name invalid: File name must comply with the following S3 object naming rules: "
            "1) Length 1-1024 bytes; "
            "2) Only letters, numbers, dots (.), underscores (_), hyphens (-), directory separators (/) are allowed; "
            "3) Spaces or the following special characters are not allowed: ? # & % { } ^ [ ] ` \\ < > ~ | \" ' + = : ; ; "
            "4) Does not start or end with /, and does not contain consecutive //; "
            "Example: report_2025-12-11.pdf, images/photo-01.png."
        )

        if not name or not name.strip():
            raise ValueError(msg + " (Reason: File name is empty)")

        # S3 limits object key to maximum 1024 bytes, apply same limit to input file name
        if len(name.encode("utf-8")) > 1024:
            raise ValueError(msg + " (Reason: Length exceeds 1024 bytes)")

        if name.startswith("/") or name.endswith("/"):
            raise ValueError(msg + " (Reason: Starts or ends with /)")
        if "//" in name:
            raise ValueError(msg + " (Reason: Contains consecutive //)")

        # Allowed character set validation
        if not FILE_NAME_ALLOWED_RE.match(name):
            bad = re.findall(r"[^A-Za-z0-9._\-/]", name)
            example = bad[0] if bad else "illegal character"
            raise ValueError(msg + f" (Reason: Contains illegal character, e.g.: {example})")

    def upload_file(self, *, file_content: bytes, file_name: str, content_type: str = "application/octet-stream", bucket: Optional[str] = None) -> str:
        # Validate input file name first to avoid generating invalid object keys
        self._validate_file_name(file_name)
        try:
            client = self._get_client()
            object_key = self._generate_object_key(original_name=file_name)
            target_bucket = self._resolve_bucket(bucket)
            client.put_object(Bucket=target_bucket, Key=object_key, Body=file_content, ContentType=content_type)
            return object_key
        except Exception as e:
            logger.error(self._error_msg("Error uploading file to S3", e))
            raise e

    def delete_file(self, *, file_key: str, bucket: Optional[str] = None) -> bool:
        try:
            client = self._get_client()
            target_bucket = self._resolve_bucket(bucket)
            client.delete_object(Bucket=target_bucket, Key=file_key)
            return True
        except Exception as e:
            logger.error(self._error_msg("Error deleting file from S3", e))
            raise e

    def file_exists(self, *, file_key: str, bucket: Optional[str] = None) -> bool:
        try:
            client = self._get_client()
            target_bucket = self._resolve_bucket(bucket)
            client.head_object(Bucket=target_bucket, Key=file_key)
            return True
        except ClientError as e:
            code = (e.response or {}).get("Error", {}).get("Code", "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            logger.error(self._error_msg("Error checking file existence in S3", e))
            return False
        except Exception as e:
            logger.error(self._error_msg("Error checking file existence in S3", e))
            return False

    def read_file(self, *, file_key: str, bucket: Optional[str] = None) -> bytes:
        try:
            client = self._get_client()
            target_bucket = self._resolve_bucket(bucket)
            resp = client.get_object(Bucket=target_bucket, Key=file_key)
            body = resp.get("Body")
            if body is None:
                raise RuntimeError("S3 get_object returned no Body")
            try:
                return body.read()
            finally:
                try:
                    body.close()
                except Exception as ce:
                    # Resource close failure doesn't affect read result, only record for debugging
                    logger.debug("Failed to close S3 response body: %s", ce)
        except Exception as e:
            logger.error(self._error_msg("Error reading file from S3", e))
            raise e

    def list_files(self, *, prefix: Optional[str] = None, bucket: Optional[str] = None, max_keys: int = 1000, continuation_token: Optional[str] = None) -> ListFilesResult:
        """List objects, supports prefix filter and pagination; returns keys/is_truncated/next_continuation_token."""
        try:
            client = self._get_client()
            target_bucket = self._resolve_bucket(bucket)
            if max_keys <= 0 or max_keys > 1000:
                raise ValueError("max_keys must be between 1 and 1000")

            kwargs: Dict[str, Any] = {
                "Bucket": target_bucket,
                "MaxKeys": max_keys,
                "Prefix": prefix,
                "ContinuationToken": continuation_token,
            }
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            resp = client.list_objects_v2(**kwargs)
            contents = resp.get("Contents", []) or []
            keys: List[str] = [item.get("Key") for item in contents if isinstance(item, dict) and item.get("Key")]
            return {
                "keys": keys,
                "is_truncated": bool(resp.get("IsTruncated")),
                "next_continuation_token": resp.get("NextContinuationToken"),
            }
        except ClientError as e:
            code = (e.response or {}).get("Error", {}).get("Code", "")
            logger.error(self._error_msg(f"Error listing files in S3 (code={code})", e))
            raise e
        except Exception as e:
            logger.error(self._error_msg("Error listing files in S3", e))
            raise e

    def generate_presigned_url(self, *, key: str, bucket: Optional[str] = None, expire_time: int = 1800) -> str:
        """Generate presigned URL through S3 proxy."""
        import json
        import urllib.request as urllib_request
        try:
            from coze_workload_identity import Client as CozeClient
            coze_client = CozeClient()
            try:
                token = coze_client.get_access_token()
            finally:
                try:
                    coze_client.close()
                except Exception:
                    # Resource release failure doesn't affect subsequent flow
                    pass
        except Exception as e:
            logger.error(f"Error loading x-storage-token: {e}")
            raise RuntimeError(f"Failed to get x-storage-token: {e}")
        try:
            sign_base = os.environ.get("COZE_BUCKET_ENDPOINT_URL") or self.endpoint_url
            if not sign_base:
                raise ValueError("Signature endpoint not configured: please set COZE_BUCKET_ENDPOINT_URL or provide endpoint_url")
            sign_url_endpoint = sign_base.rstrip("/") + "/sign-url"

            headers = {
                "Content-Type": "application/json",
                "x-storage-token": token,
            }

            target_bucket = self._resolve_bucket(bucket)
            payload = {"bucket_name": target_bucket, "path": key, "expire_time": expire_time}
            data = json.dumps(payload).encode("utf-8")
            request = urllib_request.Request(sign_url_endpoint, data=data, headers=headers, method="POST")
        except Exception as e:
            logger.error(f"Error creating request for sign-url: {e}")
            raise RuntimeError(f"Failed to create sign-url request: {e}")

        try:
            with urllib_request.urlopen(request) as resp:
                resp_bytes = resp.read()
                content_type = resp.headers.get("Content-Type", "")
                text = resp_bytes.decode("utf-8", errors="replace")
                if "application/json" in content_type or text.strip().startswith("{"):
                    try:
                        obj = json.loads(text)
                    except Exception:
                        return text
                    data = obj.get("data")
                    if isinstance(data, dict) and "url" in data:
                        return data["url"]
                    url_value = obj.get("url") or obj.get("signed_url") or obj.get("presigned_url")
                    if url_value:
                        return url_value
                    raise ValueError("Signature service response missing data.url/url field")
                return text
        except Exception as e:
            raise RuntimeError(f"GenerateSignatureURLFail: {e}")

    def stream_upload_file(
            self,
            *,
            fileobj,
            file_name: str,
            content_type: str = "application/octet-stream",
            bucket: Optional[str] = None,
            multipart_chunksize: int = 5 * 1024 * 1024,
            multipart_threshold: int = 5 * 1024 * 1024,
            max_concurrency: int = 1,
            use_threads: bool = False,
    ) -> str:
        """Streaming upload (file object)
        - fileobj: Any file-like object with read() method (e.g. open(..., 'rb'), io.BytesIO, etc.)
        - file_name: Original file name, used to generate unique key
        - content_type: MIME type
        - bucket: Object bucket; defaults to environment variable or instance default
        - multipart_chunksize: Part size (default 5MB, to adapt to proxy layer limits)
        - multipart_threshold: Threshold to trigger multipart upload (default 5MB)
        - max_concurrency: Concurrent multipart upload count (default 1, to avoid proxy layer flow control)
        - use_threads: Whether to enable thread concurrency (default False)
        Returns: Final written object key
        """
        try:
            client = self._get_client()
            target_bucket = self._resolve_bucket(bucket)
            key = self._generate_object_key(original_name=file_name)

            extra_args = {"ContentType": content_type} if content_type else {}
            # Use boto3 high-level method to execute multipart upload (pass TransferConfig to control part size)

            config = TransferConfig(
                multipart_chunksize=multipart_chunksize,
                multipart_threshold=multipart_threshold,
                max_concurrency=max_concurrency,
                use_threads=use_threads,
            )
            client.upload_fileobj(Fileobj=fileobj, Bucket=target_bucket, Key=key, ExtraArgs=extra_args, Config=config)
            return key
        except Exception as e:
            logger.error(self._error_msg("Error streaming upload (fileobj) to S3", e))
            raise e

    def upload_from_url(
            self,
            *,
            url: str,
            bucket: Optional[str] = None,
            timeout: int = 30,
    ) -> str:
        """Streaming download from URL and upload to S3
        - url: Source file URL
        - bucket: Object bucket; defaults to environment variable or instance default
        - timeout: HTTP request timeout (seconds, default 30)
        Returns: Final written object key
        """
        import urllib.request as urllib_request
        from urllib.parse import urlparse, unquote
        try:
            request = urllib_request.Request(url)
            with urllib_request.urlopen(request, timeout=timeout) as resp:
                parsed = urlparse(url)
                file_name = Path(unquote(parsed.path)).name or "file"
                content_type = resp.headers.get("Content-Type", "application/octet-stream")
                return self.stream_upload_file(
                    fileobj=resp,
                    file_name=file_name,
                    content_type=content_type,
                    bucket=bucket,
                )
        except Exception as e:
            logger.error(self._error_msg("Error uploading from URL to S3", e))
            raise e

    def trunk_upload_file(self, *, chunk_iter: Iterable[bytes], file_name: str,
                           content_type: str = "application/octet-stream", bucket: Optional[str] = None,
                           part_size: int = 5 * 1024 * 1024) -> str:
        """Streaming upload (byte iterator, explicit multipart upload)
        - chunk_iter: Iterable object, produces bytes chunk by chunk; each chunk size is variable (accumulated internally to part_size before upload), last chunk may be less than 5MB
        - file_name: Original file name, used to generate unique key
        - content_type: MIME type
        - bucket: Object bucket; defaults to environment or instance default
        - part_size: Minimum size for each part (except the last one); default 5MB
        Returns: Final written object key
        """
        client = self._get_client()
        target_bucket = self._resolve_bucket(bucket)
        key = self._generate_object_key(original_name=file_name)

        # Initialize multipart upload
        try:
            init_resp = client.create_multipart_upload(Bucket=target_bucket, Key=key, ContentType=content_type)
            upload_id = init_resp["UploadId"]
        except Exception as e:
            logger.error(self._error_msg("create_multipart_upload failed", e))
            raise e

        parts = []
        part_number = 1
        buffer = bytearray()
        try:
            for chunk in chunk_iter:
                if not chunk:
                    continue
                buffer.extend(chunk)
                while len(buffer) >= part_size:
                    data = bytes(buffer[:part_size])
                    buffer = buffer[part_size:]
                    resp = client.upload_part(Bucket=target_bucket, Key=key, UploadId=upload_id, PartNumber=part_number,
                                              Body=data)
                    parts.append({"PartNumber": part_number, "ETag": resp["ETag"]})
                    part_number += 1

            # Upload remaining data less than part_size
            if len(buffer) > 0:
                resp = client.upload_part(Bucket=target_bucket, Key=key, UploadId=upload_id, PartNumber=part_number,
                                          Body=bytes(buffer))
                parts.append({"PartNumber": part_number, "ETag": resp["ETag"]})

            # Complete multipart upload
            client.complete_multipart_upload(
                Bucket=target_bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
            )
            return key
        except Exception as e:
            logger.error(self._error_msg("multipart upload failed", e))
            try:
                client.abort_multipart_upload(Bucket=target_bucket, Key=key, UploadId=upload_id)
            except Exception as ae:
                logger.error(self._error_msg("abort_multipart_upload failed", ae))
            raise e
