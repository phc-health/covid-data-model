#!/bin/bash
# deploy-docs.sh - Deploys API Documentation

set -o nounset
set -o errexit

CMD=$0

# Checks command-line arguments, sets variables, etc.
prepare () {

  if [[ -z ${GITHUB_TOKEN:-} ]]; then
    echo "Error: GITHUB_TOKEN must be set to a personal access token. See:"
    echo "https://help.github.com/en/github/authenticating-to-github/creating-a-personal-access-token-for-the-command-line"
    exit 1
  fi
}

execute () {
  curl -H "Authorization: token $GITHUB_TOKEN" \
       -H "Accept: application/vnd.github.v3+json" \
      --request POST \
      --data "{ \"ref\": \"main\" }" \
      https://api.github.com/repos/covid-projections/covid-data-model/actions/workflows/deploy_docs.yml/dispatches

  echo "Deploying API Documentation. Go to https://github.com/covid-projections/covid-data-model/actions to monitor progress."
}

prepare "$@"
execute
