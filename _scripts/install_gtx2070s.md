```
wget -o driver.run https://www.nvidia.com/content/DriverDownload-March2009/confirmation.php?url=/XFree86/Linux-x86_64/440.44/NVIDIA-Linux-x86_64-440.44.run&lang=us&type=TITAN

chmod +x driver.run

sudo apt-get install gcc

sudo apt-get install build-essential

sudo ./driver.run

```

Test driver installation using

```
nvidia-smi
```

Install CUDA

```
wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run

export PATH=$PATH:/usr/local/cuda-10.2/bin
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64

echo 'PATH=$PATH:/usr/local/cuda-10.2/bin' >> ~/.bash_profile
echo 'LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64' >> ~/.bash_profile

```

Install CUDnn

```
sudo dpkg -i libcudnn7_7.3.0.29–1+cuda10.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.3.0.29–1+cuda10.0_amd64.deb
```

Install bazel

``` 
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get install bazel
```