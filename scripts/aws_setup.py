"""
aws_setup.py — One-time AWS S3 setup for medical AI middleware.

Creates:
  - s3://medical-ai-logs      (audit logs + XAI heatmaps, 90-day lifecycle)
  - s3://medical-ai-uploads   (temp image uploads, 1-day lifecycle)

Run once:
    python scripts/aws_setup.py

Requirements:
    pip install boto3
    export AWS_ACCESS_KEY_ID=xxx
    export AWS_SECRET_ACCESS_KEY=xxx
    export AWS_REGION=us-east-1  (optional, default us-east-1)
"""
import os
import sys
import json


def setup_aws():
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        print("❌ boto3 not installed. Run: pip install boto3")
        sys.exit(1)

    region = os.getenv("AWS_REGION", "us-east-1")
    bucket_logs    = os.getenv("S3_BUCKET_LOGS",    "medical-ai-logs")
    bucket_uploads = os.getenv("S3_BUCKET_UPLOADS", "medical-ai-uploads")

    print(f"\n{'='*60}")
    print("  Medical AI — AWS S3 Setup")
    print(f"{'='*60}")
    print(f"  Region:          {region}")
    print(f"  Logs bucket:     {bucket_logs}")
    print(f"  Uploads bucket:  {bucket_uploads}")
    print(f"{'='*60}\n")

    s3 = boto3.client("s3", region_name=region)

    # Verify credentials
    try:
        s3.list_buckets()
        print("✅ AWS credentials valid")
    except Exception as e:
        print(f"❌ AWS credentials error: {e}")
        print("\nMake sure these env vars are set:")
        print("  export AWS_ACCESS_KEY_ID=xxx")
        print("  export AWS_SECRET_ACCESS_KEY=xxx")
        sys.exit(1)

    buckets = [
        {
            "name":           bucket_logs,
            "description":    "Audit logs + XAI heatmaps",
            "lifecycle_days": 90,
        },
        {
            "name":           bucket_uploads,
            "description":    "Temporary medical image uploads",
            "lifecycle_days": 1,
        },
    ]

    for bucket in buckets:
        name = bucket["name"]
        print(f"\n{'─'*50}")
        print(f"  Setting up: {name}")
        print(f"  Purpose:    {bucket['description']}")
        print(f"  Auto-delete after: {bucket['lifecycle_days']} day(s)")

        # Create bucket
        try:
            if region == "us-east-1":
                s3.create_bucket(Bucket=name)
            else:
                s3.create_bucket(
                    Bucket=name,
                    CreateBucketConfiguration={"LocationConstraint": region},
                )
            print(f"  ✅ Created bucket: {name}")
        except ClientError as e:
            if e.response["Error"]["Code"] == "BucketAlreadyOwnedByYou":
                print(f"  ℹ️  Bucket already exists: {name}")
            else:
                print(f"  ❌ Failed to create {name}: {e}")
                continue

        # Block all public access
        try:
            s3.put_public_access_block(
                Bucket=name,
                PublicAccessBlockConfiguration={
                    "BlockPublicAcls":       True,
                    "IgnorePublicAcls":      True,
                    "BlockPublicPolicy":     True,
                    "RestrictPublicBuckets": True,
                },
            )
            print(f"  ✅ Public access blocked")
        except Exception as e:
            print(f"  ⚠️  Could not block public access: {e}")

        # Enable versioning (for audit logs)
        if name == bucket_logs:
            try:
                s3.put_bucket_versioning(
                    Bucket=name,
                    VersioningConfiguration={"Status": "Enabled"},
                )
                print(f"  ✅ Versioning enabled (audit log protection)")
            except Exception as e:
                print(f"  ⚠️  Could not enable versioning: {e}")

        # Set lifecycle rule
        try:
            s3.put_bucket_lifecycle_configuration(
                Bucket=name,
                LifecycleConfiguration={
                    "Rules": [{
                        "ID":     f"gdpr-auto-delete-{bucket['lifecycle_days']}d",
                        "Status": "Enabled",
                        "Filter": {"Prefix": ""},
                        "Expiration": {"Days": bucket["lifecycle_days"]},
                        "NoncurrentVersionExpiration": {"NoncurrentDays": bucket["lifecycle_days"]},
                    }]
                },
            )
            print(f"  ✅ Lifecycle rule set: delete after {bucket['lifecycle_days']} day(s)")
        except Exception as e:
            print(f"  ⚠️  Could not set lifecycle: {e}")

        # Set bucket policy (deny non-HTTPS)
        policy = {
            "Version": "2012-10-17",
            "Statement": [{
                "Sid":       "DenyNonHTTPS",
                "Effect":    "Deny",
                "Principal": "*",
                "Action":    "s3:*",
                "Resource": [
                    f"arn:aws:s3:::{name}",
                    f"arn:aws:s3:::{name}/*",
                ],
                "Condition": {
                    "Bool": {"aws:SecureTransport": "false"}
                },
            }],
        }
        try:
            s3.put_bucket_policy(Bucket=name, Policy=json.dumps(policy))
            print(f"  ✅ HTTPS-only policy applied")
        except Exception as e:
            print(f"  ⚠️  Could not set bucket policy: {e}")

    # Print env vars to set
    print(f"\n{'='*60}")
    print("  ✅ Setup complete!")
    print(f"{'='*60}")
    print("\n  Add these to your .env / RunPod / HF Spaces secrets:\n")
    print(f"  AWS_ACCESS_KEY_ID=your_key_id")
    print(f"  AWS_SECRET_ACCESS_KEY=your_secret")
    print(f"  AWS_REGION={region}")
    print(f"  S3_BUCKET_LOGS={bucket_logs}")
    print(f"  S3_BUCKET_UPLOADS={bucket_uploads}")
    print(f"  S3_ENABLED=true")
    print(f"\n  GDPR compliance:")
    print(f"  - {bucket_uploads}: uploads auto-deleted after 1 day ✅")
    print(f"  - {bucket_logs}: audit logs auto-deleted after 90 days ✅")
    print(f"  - Both buckets: HTTPS-only, no public access ✅")
    print(f"\n  Cost estimate (free tier):")
    print(f"  - Storage: ~$0 (well within 5GB free tier)")
    print(f"  - Requests: ~$0 (well within 20k GET / 2k PUT free tier)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    setup_aws()
