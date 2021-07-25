# dog-fan

# Hardware
- RPi 3 B+
- RPi Camera
- TP-Link/KASA WiFi SmartPlug

# Description
A simple program to use computer vision to identify if my dog is sitting in front of her favorite fan.


# Download models
Using download and requirements script from TensorFlow Examples

chmod +x ./download.sh
./download.sh models/

# Run program
python3 dog_fan.py --model models/detect.tflite --labels models/coco_labels.txt
