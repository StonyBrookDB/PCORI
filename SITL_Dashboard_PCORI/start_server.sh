#!/bin/bash
# Start the PCORI SITL Prototype server
# Migrated from /home/yinan/pcori_prototype
# Updated: December 27, 2024

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Export the OpenAI API key from environment
# Using naming convention from key file: OPENAI_API_KEY_FS_112425
export OPENAI_API_KEY="${OPENAI_API_KEY_FS_112425}"

# Export the Anthropic API key for Claude sandbox analysis
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}"

# Add packages to PYTHONPATH for yl.job_system and other yl modules
export PYTHONPATH="/home/yinan/checkouts/v0/packages:${PYTHONPATH}"

# Start uvicorn
# Assumes eliu3 conda environment is active (python will be from eliu3)
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8501
