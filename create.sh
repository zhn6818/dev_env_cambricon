sudo docker run -it --cap-add=sys_ptrace \
        --device /dev/cambricon_ctl \
        --device /dev/cambricon_dev0 \
        --device /dev/cambricon_ipcm0 \
        -v /sys/kernel/debug:/sys/kernel/debug \
        -v /dev/cambricon:/dev/cambricon \
        -v /mnt/:/mnt/ \
        -v /usr/bin/cnmon:/usr/bin/cnmon\
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v /opt:/opt\
        -v /data:/data\
        -v /data1:/data1\
        --shm-size="64g"\
        -p 12237:22\
        --name zhn_cam \
        yellow.hub.cambricon.com/magicmind/release/x86_64/magicmind:1.0.1-x86_64-ubuntu18.04-py_3_7 /bin/bash
