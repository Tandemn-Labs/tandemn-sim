#!/usr/bin/env bash
set -Eeuo pipefail

bucket=${S3_BUCKET:-batched-chunks}
region=us-east-2

command -v aws >/dev/null || {
    printf 'aws CLI is required\n' >&2
    exit 1
}

for job_id in \
    0000000000RTXPS9V40NCX15CX \
    0000000000QWKV2ZQMRNCDQA0W \
    0000000000GZQ6NH3WGDFAB181 \
    0000000000SYGGTBGSR1ANV835 \
    00000000000K6RCQQS2D8XK2YA
do
    key_count=$(aws --region "$region" s3api list-objects-v2 \
        --bucket "$bucket" \
        --prefix "${job_id}/output/" \
        --max-keys 1 \
        --query KeyCount \
        --output text)
    [[ "$key_count" == 0 ]] && continue
    # S3 has no real directories; deleting this recursive prefix removes output/ itself.
    aws --region "$region" s3 rm "s3://${bucket}/${job_id}/output/" --recursive
    printf '%s\n' "$job_id"
done
