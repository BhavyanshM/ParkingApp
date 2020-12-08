#! /bin/sh

python3 parking.py </dev/null &>/dev/null &
echo "Waiting to Initialize for 20 Seconds"
sleep 20
python3 parking_server.py

