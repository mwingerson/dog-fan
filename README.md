# Dog Fan

# Hardware
- RPi 3 B+
- RPi Camera
- TP-Link/KASA WiFi SmartPlug

# 3D models used
stand parts 1 - https://www.thingiverse.com/thing:3114849
stand parts 2 - https://www.prusaprinters.org/prints/7867-simple-stand-for-sneakss-articulating-rpi-camera-m
RPi case - https://www.thingiverse.com/thing:1956623

# Description
A simple program to use computer vision to identify if my dog is sitting in front of her favorite fan.


# Download models
Using download and requirements script from TensorFlow Examples

chmod +x ./download.sh
./download.sh models/

# Run program
python3 dog_fan.py --model models/detect.tflite --labels models/coco_labels.txt

OR

python3 dog_fan.py 
