sudo ifconfig wlan0 up
sudo wpa_supplicant -c /etc/wpa_supplicant.conf -i wlan0 &
sudo dhclient wlan0
