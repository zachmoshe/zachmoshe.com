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


if [ "$env" == "development" ]; then
	aws_command="aws --profile zachmoshe.com s3 sync _site s3://zachmoshe.com-drafts"
else
	aws_command="aws --profile zachmoshe.com s3 sync _site s3://zachmoshe.com"
fi

echo "Run this command to sync with S3:"
echo $aws_command
