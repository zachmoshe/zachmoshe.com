#! /bin/bash

env=$1

if [ -z "$env" ]; then
	env="development"
fi

if [ "$env" != "production" ] && [ "$env" != "development" ]; then
	echo "env must be 'production' or 'development' (or null)"
	exit 1
fi

echo JEKYLL_ENV=$env
JEKYLL_ENV=$env bundle exec jekyll build

# I have no idea how to get rid of this file
rm -fr _site/assets/.sprockets-manifest-*.json


if [ "$env" == "development" ]; then
	target_bucket="s3://zachmoshe.com-drafts"
else 
	target_bucket="s3://zachmoshe.com"
fi

echo "Run this command to sync with S3:"
echo "aws --profile zachmoshe.com s3 sync _site $target_bucket --delete --exclude 'assets/*'"
echo "aws --profile zachmoshe.com s3 sync _site $target_bucket --delete --include 'assets/*' --cache-control max-age=31536000"
