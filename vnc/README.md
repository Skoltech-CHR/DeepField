# **GUI remote rendering with VirtualGl + VNC on Ubuntu Server (debian-based server)**

### Example requisites for installation: 
Server example IP: **145.15.42.45** \
Server hostname: **REMOTESRV** \
Username on server: **testuser** \
Password for user: **testpass** \
Password for VNC access: **vncpass**

## Configuring server:
### Install prerequisites:
```
sudo -i
# Openg GL necessary for Virtual GL:
apt-get install freeglut3-dev mesa-utils
# Install xserver (or reinstall):
apt-get --reinstall install xserver-xorg-core
# Extra packages for xserver:
 apt install x11-xserver-utils libxrandr-dev
# Install lightdm:
apt-get install lightdm
# Install nvidia-drivers
service lightdm stop
bash ./NVIDIA-Linux-x86_64-460.73.01.run --no-cc-version-check
```

### Configure Xorg and turn off HardDPMS for nvidia cards:
Run `nvidia-xconfig --query-gpu-info` to obtain the bus ID of the GPU. Example:
```
GPU #0:
  Name      : Tesla M60
  ...
  PCI BusID : PCI:136:0:0
```
Create an appropriate **xorg.conf** file for headless operation:
```
sudo nvidia-xconfig -a --allow-empty-initial-configuration --virtual=1920x1200 --busid {busid}
```
Note: Foe headless server use additional flag: ```--use-display-device=None```

Replace *{busid}* with the bus ID you obtained in Step 1. 

If you are using version 440.xx or later of the nVidia proprietary driver, then edit **xorg.conf** and add
```
Option "HardDPMS" "false"
```
under the Device or Screen section.

### Install VirtualGL:
```
# Login as super user
sudo -i
# Stop lightdm service
service lightdm stop
# Download VirtualGL:
wget https://downloads.sourceforge.net/project/virtualgl/2.6.5/virtualgl_2.6.5_amd64.deb
# Install VirtualGL:
dpkg -i virtualgl_2.6.5_amd64.deb
# Turn off nvidia services:
rmmod nvidia_drm
rmmod nvidia_modeset
rmmod nvidia_uvm
rmmod nvidia
# Run VirtualGL configuration:
/opt/VirtualGL/bin/vglserver_config
# Note: For all questions answer "Yes". 
```

### Installing VirtualGL creates new group "vglusers", so to use VirtualGL features we need to assosiate user with this group:
```
usermod -a -G vglusers testuser
```

### Check whether installation is correct:
```
xauth merge /etc/opt/VirtualGL/vgl_xauth_key
xdpyinfo -display :0
/opt/VirtualGL/bin/glxinfo -display :0 -c
```

### Install TurboVNC:
```
# Download TurboVNC:
wget https://downloads.sourceforge.net/project/turbovnc/2.2.6/turbovnc_2.2.6_amd64.deb
# Install TurboVNC:
dpkg -i turbovnc_2.2.6_amd64.deb
# Exit sudo mode:
exit
```

## Forwarding graphics:
### Run TurboVNC server with command:
```
/opt/TurboVNC/bin/vncserver
```
> Note: 'vncserver' created a display that it will translate, after running the command, it will ask to set a password (e.g. 'vncpass' as password. Password can be reset by deleting file ~/.vnc/passwd) and created display with name "REMOTESRV:0" (or REMOTESRV:1, REMOTESRV:2, etc.). We need the number after server hostname. The other way to get this display name is to run command:
```
/opt/TurboVNC/bin/vncserver -list 
```

### Set display environment as selected display. E.g. if vncserver creted "REMOTESRV:2" display, set:
```
export DISPLAY=:2
```

### After this step our server is set up. Let's connect to it remotely. Server creates display accessible by 5900 + {n} port, where n - is the number of display, selected by vncserver. In our example DISPLAY=2 so the port will be 5902. Connect to server by ssh with username/password (testuser/testpass for our example).
```
ssh testuser@145.15.42.45 -L 5902:localhost:5902
```
> Reminder: IP 145.15.42.45 is an example IP. User you server IP or hostname.
> Note: this command will forward server port 5902 to your PC local port 5902

### Install Turbo VNC Viewer on your machine (https://sourceforge.net/projects/turbovnc/files/2.2.6/)

### Run Turbo VNC Viewer and set as "VNC server" the following:
```
"VNC server": localhost:5902
Password:     vncpass
```

### After succesfull login you will see linux GUI and login screen. To enter set yout login/pass for the system, e.g.:
```
Login:    testuser
Password: testpass
```

### How to run applications or scripts that uses VTK:
You should run any app or script using ```vglrun``` command, e.g. (remember ser propriate $DISPLAY variable value w.r.t. vncserver display):
```
vglrun glxgears
vglrun python show.py
vglrun gdb app
```
