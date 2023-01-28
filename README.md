# dev_env_cambricon



## Dependence

* ubuntu18.04
* [MLU370-S4智能加速卡](https://www.cambricon.com/index.php?m=content&c=index&a=lists&catid=365)


## Installation

* [driver tookit eg...](https://pan.baidu.com/s/1A9tNymLKwbxeTnx5ATSPQg?pwd=67ir)     密码：67ir

```
#检查设备资源是否分配成功
lspci -v -d:0370
```

<div align="center">
    <a href="./">
        <img src="./figure/lspci.png" width="79%"/>
    </a>
</div>

这里选择的驱动版本是 cambricon-mlu-driver-ubuntu18.04-dkms_4.20.18_amd64.deb

[寒武纪ubunt安装方式有两种，这里使用dpkg，另一种使用apt安装](https://www.cambricon.com/docs/sdk_1.9.0/driver_4.20.18/user_guide_4.20.18/driver_3/driver_3.html#ubuntudebian)

``` shell
sudo dpkg -i cambricon-mlu-driver-ubuntu18.04-dkms_4.20.18_amd64.deb
```
``` shell
cnmon
```
<div align="center">
    <a href="./">
        <img src="./figure/cnmon.png" width="79%"/>
    </a>
</div>

* cnmon成功则说明驱动安装成功

* docker安装(version>=v19.03)

``` shell
docker --version
```
<div align="center">
    <a href="./">
        <img src="./figure/docker_version.png" swidth="79%"/>
    </a>
</div>


* 在百度网盘中下载magicmind_1.0.1-1_ubuntu18.04.tar.gz驱动

``` shell
sudo docker load -i magicmind_1.0.1-1_ubuntu18.04.tar.gz

sudo docker images
```

``` shell

sudo docker run -it --cap-add=sys_ptrace \
        --device /dev/cambricon_ctl \   #--device 寒武纪设备必须
        --device /dev/cambricon_dev0 \
        --device /dev/cambricon_ipcm0 \
        -v /sys/kernel/debug:/sys/kernel/debug \
        -v /dev/cambricon:/dev/cambricon \   #-v path to path
        -v /mnt/:/mnt/ \
        -v /usr/bin/cnmon:/usr/bin/cnmon\
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v /opt:/opt\
        -v /data:/data\
        -v /data1:/data1\
        --shm-size="64g"\
        -p 12237:22\      #port
        --name zhn_cam \  #container name
        yellow.hub.cambricon.com/magicmind/release/x86_64/magicmind:1.0.1-x86_64-ubuntu18.04-py_3_7 /bin/bash
```
## 使用教程

``` shell
 mkdir build
 cd build
 cmake .. -DCMAKE_INSTALL_PREFIX=../install/
 make
```

## 支持的模型

[model教程](./model/)
