# Install NVIDIA driver

## Official documentation
  Please refer to the [official NVIDIA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

## Step by step instructions

**The instructions below worked for us, but different versions may be required for your system**

Install GCC compiler & other tools:
```bash
sudo apt update
sudo apt install build-essential
sudo apt-get install software-properties-common
```

Add the following two package sources to /etc/apt/sources.list and then update apt:
```bash
deb http://deb.debian.org/debian/ bullseye main contrib non-free
deb-src http://deb.debian.org/debian/ bullseye main contrib non-free
sudo apt-get update
sudo apt update
```

Install nvidia toolkit:
```bash
sudo apt-get install linux-headers-$(uname -r)
sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit
sudo reboot
```

After system reboots, install nvidia driver:
```bash
export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
sudo apt-get install nvidia-open
```

To check that everything installed properly, run this command to see the smi interface:
```bash
nvidia-smi
```

