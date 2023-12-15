## Environment setup for Intel Arc GPU
For Linux users, Ubuntu 22.04 and Linux kernel 5.19.0 is prefered. Ubuntu 22.04 and Linux kernel 5.19.0-41-generic is mostly used in our test environment. But default linux kernel of ubuntu 22.04.3 is 6.2.0-35-generic, so we recommonded you to downgrade kernel to 5.19.0-41-generic to archive the best performance. 
### 1. Downgrade kernels
Here are the steps to downgrade your kernel:
```bash
# downgrade kernel to 5.19.0-41-generic
  
sudo apt-get update && sudo apt-get install  -y --install-suggests  linux-image-5.19.0-41-generic

sudo sed -i "s/GRUB_DEFAULT=.*/GRUB_DEFAULT=\"1> $(echo $(($(awk -F\' '/menuentry / {print $2}' /boot/grub/grub.cfg \
| grep -no '5.19.0-41' | sed 's/:/\n/g' | head -n 1)-2)))\"/" /etc/default/grub

sudo  update-grub

sudo reboot
# As 5.19's kernel doesn't has any arc graphic driver. The machine may not start the desktop correctly, but we can use the ssh to login. 
# Or you can select 5.19's recovery mode in the grub, then choose resume to resume the normal boot directly.
```
**Notice:  As 5.19's kernel doesn't has any arc graphic driver. The machine may not start the desktop correctly, but we can use the ssh to login. Or you can select 5.19's recovery mode in the grub, then choose resume to resume the normal boot directly.**  
You can remove the 6.2.0 kernel if you don't need it. It's an optional step.
```bash 
# remove latest kernel (optional)
sudo apt purge linux-image-6.2.0-*
sudo apt autoremove
sudo reboot
```

#### 2. Install GPU driver
Here is the steps to install gpu driver:
```bash
# install drivers
# setup driver's apt repository
sudo apt-get install -y gpg-agent wget
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list

sudo apt-get update

sudo apt-get -y install \
    gawk \
    dkms \
    linux-headers-$(uname -r) \
    libc6-dev
	
sudo apt install intel-i915-dkms=1.23.5.19.230406.21.5.17.0.1034+i38-1 intel-platform-vsec-dkms=2023.20.0-21 intel-platform-cse-dkms=2023.11.1-36 intel-fw-gpu=2023.39.2-255~22.04

sudo apt-get install -y gawk libc6-dev udev\
  intel-opencl-icd intel-level-zero-gpu level-zero \
  intel-media-va-driver-non-free libmfx1 libmfxgen1 libvpl2 \
  libegl-mesa0 libegl1-mesa libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri \
  libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm12 libxatracker2 mesa-va-drivers \
  mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all vainfo
  
sudo reboot

# Configuring permissions

sudo gpasswd -a ${USER} render
newgrp render

# Verify the device is working with i915 driver
sudo apt-get install -y hwinfo
hwinfo --display
```

### 3. Install OneAPI and BigDL
```
# config OneAPI repository
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
```
Torch 2.0 and Torch 2.1 need different oneapi version, so we need to install different OneAPI.  
**Torch 2.1** wants the latest oneapi, you can install like this:
```bash
sudo apt install -y intel-basekit # for torch 2.1 and ipex 2.1
# install bigdl
conda create -n llm python=3.9
conda activate llm
pip install --pre --upgrade bigdl-llm[xpu_2.1] -f https://developer.intel.com/ipex-whl-stable-xpu
```
**Torch 2.0** wants the latest oneapi, you can install like this:
```
sudo apt install -y intel-oneapi-common-vars=2023.2.0-49462 \
    intel-oneapi-compiler-cpp-eclipse-cfg=2023.2.0-49495 intel-oneapi-compiler-dpcpp-eclipse-cfg=2023.2.0-49495 \
    intel-oneapi-diagnostics-utility=2022.4.0-49091 \
    intel-oneapi-compiler-dpcpp-cpp=2023.2.0-49495 \
    intel-oneapi-mkl=2023.2.0-49495 intel-oneapi-mkl-devel=2023.2.0-49495 \
    intel-oneapi-mpi=2021.10.0-49371 intel-oneapi-mpi-devel=2021.10.0-49371 \
    intel-oneapi-tbb=2021.10.0-49541 intel-oneapi-tbb-devel=2021.10.0-49541\
    intel-oneapi-ccl=2021.10.0-49084 intel-oneapi-ccl-devel=2021.10.0-49084\
    intel-oneapi-dnnl-devel=2023.2.0-49516 intel-oneapi-dnnl=2023.2.0-49516
# install bigdl
conda create -n llm python=3.9
conda activate llm
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
```
