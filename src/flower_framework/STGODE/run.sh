#!/bin/bash


echo "Starting server"

source ../../../project_env/bin/activate                    

python3 fed_stgode.py       

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
