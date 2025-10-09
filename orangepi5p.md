```
sudo apt update
sudo apt install git build-essential bison flex libssl-dev dwarves

git clone https://github.com/orangepi-xunlong/orangepi-build.git
cd orangepi-build
./build.sh >> Kernel package >> Do not change the kernel configuration >> orangepi5plus >> Current Recommended
cd output/debs
sudo dpkg -i linux-headers-current-rockchip-rk3588_1.2.0_arm64.deb

or

wget https://github.com/user-attachments/files/22798846/linux-headers-6.1.x.tar.gz
tar xzvf linux-headers-6.1.x.tar.gz
sudo dpkg -i linux-headers-current-rockchip-rk3588_1.2.0_arm64.deb
```

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
