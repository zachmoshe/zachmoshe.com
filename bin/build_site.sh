#! /bin/bash

env=$1

if [ -z "$env" ]; then
	env="development"
fi

echo JEKYLL_ENV=$env
JEKYLL_ENV=$env bundle exec jekyll build
