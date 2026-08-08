"""Presigned URL helper (this IAM user cannot create instance roles).

Usage: python aws/presign.py get <key> | put <key>  [--expires 172800]
"""

import argparse

import boto3

BUCKET = 'banino-repro-975050064729'


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('method', choices=['get', 'put'])
  ap.add_argument('key')
  ap.add_argument('--expires', type=int, default=172800)
  args = ap.parse_args()
  s3 = boto3.session.Session(profile_name='banino-repro').client('s3')
  op = 'get_object' if args.method == 'get' else 'put_object'
  print(s3.generate_presigned_url(
      op, Params={'Bucket': BUCKET, 'Key': args.key},
      ExpiresIn=args.expires))


if __name__ == '__main__':
  main()
