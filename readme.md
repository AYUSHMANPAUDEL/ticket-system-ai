# Setup guide

## installation
We will start by installing Ubuntu using wsl

Open PowerShell as Administrator and run:

```powershell
wsl --install -d Ubuntu-22.04
```

## Update Ubuntu and install build dependencies
```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y build-essential git wget curl subversion \
    libncurses5-dev libxml2-dev libsqlite3-dev uuid-dev libjansson-dev \
    libssl-dev pkg-config libedit-dev libssl-dev libopus-dev \
    libsndfile1 ffmpeg python3 python3-pip
```

## Download, compile and install Asterisk 22
```bash
cd ~
wget http://downloads.asterisk.org/pub/telephony/asterisk/asterisk-22-current.tar.gz
tar xvf asterisk-22-current.tar.gz
cd asterisk-22.6.0    # version folder may vary

./configure
make -j$(nproc)
sudo make install
sudo make samples   # installs sample config files into /etc/asterisk
sudo make config
sudo systemctl enable asterisk
sudo systemctl start asterisk
```

## to access astrix cli
```bash
sudo asterisk -rvvv
```

## check if the installation worked
```bash
sudo asterisk -rx "core show version"
```
If it prints the version, it’s running!

# Configuration

## PJSB config
```bash
sudo mv /etc/asterisk/pjsip.conf /etc/asterisk/pjsip.conf.backup
sudo nano /etc/asterisk/pjsip.conf
```
paste the codes from `pjsip.txt`

## extensions config
```bash
sudo mv /etc/asterisk/extensions.conf /etc/asterisk/extensions.conf.backup
sudo nano /etc/asterisk/extensions.conf
```
paste the content of `extensions.txt`

## http config
```bash
sudo mv /etc/asterisk/http.conf /etc/asterisk/http.conf.backup
sudo nano /etc/asterisk/http.conf
```
paste the content of http.txt!

## ari config
```bash
sudo mv /etc/asterisk/ari.conf /etc/asterisk/ari.conf.backup
sudo nano /etc/asterisk/ari.conf
```
paste the content of ari.txt 

## voicebot js prep
sudo nano /opt/voicebot/voicebot.js
paste the file content of voicebot.js

# Install Linphone
Install Linphone from the link https://www.linphone.org/en/download/

setup it up

click on manual SIP account

set domain(sip server address) from your wifi by running the command `ip add` in ubuntu (get help from chatgpt if needed)

set that as domain

usernames = 1001 , 1002

passwords = 1234, 1234





