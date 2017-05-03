#! /bin/bash

env=$1

if [ -z "$env" ]; then
	env="production"
fi

JEKYLL_ENV=$env bundle exec jekyll build
