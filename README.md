bash -c "$(wget -qLO - https://raw.githubusercontent.com/JorisOpsommer/frigate-yolo/refs/heads/main/lxc/init.sh)"

# logs

dev/shm/logs/frigate

# run frigate in dev mode

python3 -m frigate
