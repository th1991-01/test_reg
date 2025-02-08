nvidia-driver
```
#もともと入っているドライバの削除
$ sudo apt-get --purge remove nvidia-*
$ sudo apt-get --purge remove cuda-*

#nouveau exsists?
$ lsmod | grep -i nouveau

$ sudo nano /etc/modprobe.d/blacklist-nouveau.conf
blacklist nouveau
options nouveau modeset=0
# Save(Ctrl+S) Close(Ctrl+X)

$ sudo update-initramfs -u

$ ubuntu-drivers devices # check
$ sudo apt install -y nvidia-driver-550
$ reboot

##確認
$ nvidia-smi
```

cuda
```
##runfile
$ wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
$ sudo sh cuda_12.4.0_550.54.14_linux.run

## Continue -> accept -> CUDA Toolkit 12.4にチェックを入れる＋driverのチェックを外す = Install

##include pass
$ echo 'export PATH="/usr/local/cuda-12.4/bin:$PATH"' >> ~/.bashrc
$ echo 'export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
$ source ~/.bashrc

##確認
$ nvcc -V
```

anaconda
```
$ sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
$ curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
$ bash Anaconda3-2024.10-1-Linux-x86_64.sh
## Enter
## yes: Please answer 'yes' or 'no':'
## Enter: Press ENTER to confirm the location
## yes: You can undo this by running `conda init --reverse $SHELL`?

$ source ~/.bashrc
```

install
```
conda create -n hishinuma python==3.12
pip install -U "jax[cuda12]"
pip install flax

pip install scikit-learn matplotlib

pip install gymnasium==0.29.1
pip install gymnasium[mujoco]
```

test env
```
$ python -c "import jax;print(jax.devices())"
##  [CudaDevice(id=0)] が表示されればOK
```

test reg
```
python get_dataset_gym.py
python reg_jax.py
```
