#!/bin/bash
#SBATCH --job-name=vscode
#SBATCH --mem=8G
#SBATCH -c 4
#SBATCH -t 2:00:00
#SBATCH --output=vs_code.out
#SBATCH --error=vs_code.err
~/.vscode-server/bin/<commit_id>/server.sh --host 0.0.0.0 --port 8080