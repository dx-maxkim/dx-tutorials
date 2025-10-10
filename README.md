# Installation Guide

## Prerequisites:
```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install python3-venv build-essential python3-dev git-all ffmpeg
```

## Download this dx-tutorials repo
```bash
git clone https://github.com/dx-maxkim/dx-tutorials.git
cd dx-tutorials
```


## Create/Activate a python3 virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install required pip packages:
```bash
pip install -U pip
pip install -r requirements.txt
```


# Usages
## Run jupyter-lab
```bash
jupyter-lab
```

## External connection
```bash
jupyter-lab --generate-config
vi ~/.jupyter/jupyter_lab_config.py
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.token = ''
c.ServerApp.password = ''
c.ServerApp.open_browser = False
```
