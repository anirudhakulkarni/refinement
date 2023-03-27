repeatedly used commands
Visualization:
https://github.com/Jonathan-Pearce/calibration_library/blob/master/visualization.py

# kill all screen sessions that start with "52*"
screen -ls | grep 52 | cut -d. -f1 | awk '{print $1}' | xargs kill~~
