#!/usr/bin/env bash
set -Eeuo pipefail

bucket=${S3_BUCKET:-batched-chunks}
region=us-east-2

command -v aws >/dev/null || {
    printf 'aws CLI is required\n' >&2
    exit 1
}

aws --region "$region" s3 ls "s3://${bucket}/" | while read -r marker prefix; do
    job_id=${prefix%/}
    [[ "$marker" == PRE && "$job_id" =~ ^[0-9A-HJKMNP-TV-Z]{26}$ ]] || continue
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
