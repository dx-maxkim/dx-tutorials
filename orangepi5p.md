<img width="845" height="202" alt="image" src="https://github.com/user-attachments/assets/5562ca30-aaf3-4c9d-98ee-7bd6a746814e" />```
sudo apt update
sudo apt install git build-essential bison flex libssl-dev dwarves

git clone https://github.com/orangepi-xunlong/orangepi-build.git
cd orangepi-build


# version down-grade:
sudo apt -s install --allow-downgrades \
  libgstreamer-plugins-base1.0-0=1.20.1-1ubuntu0.5 \
  libgstreamer-gl1.0-0=1.20.1-1ubuntu0.5 \
  gir1.2-gst-plugins-base-1.0=1.20.1-1ubuntu0.5 \
  libgstreamer-plugins-base1.0-dev=1.20.1-1ubuntu0.5

sudo apt install -y --allow-downgrades \
  libgstreamer-plugins-base1.0-0=1.20.1-1ubuntu0.5 \
  libgstreamer-gl1.0-0=1.20.1-1ubuntu0.5 \
  gir1.2-gst-plugins-base-1.0=1.20.1-1ubuntu0.5 \
  libgstreamer-plugins-base1.0-dev=1.20.1-1ubuntu0.5

```
