## Installation Guide
```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install python3-venv build-essential python3-dev git-all ffmpeg
```

```
python3 -m venv ~/pipx-venv
source ~/pipx-venv/bin/activate
pip install pipx
pipx install uv
deactivate
uv --version
```

## Create a Virtual Environment
```
uv venv jupyter-env
source jupyter-env/bin/activate
uv pip install -r requirements.txt
```

## How to run
```
jupyter-lab
```

```
jupyter-lab --generate-config
vi ~/.jupyter/jupyter_lab_config.py
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.token = ''
c.ServerApp.password = ''
c.ServerApp.open_browser = False
```
