# python3
#
# Learning from TensorFlow object_detection
# examples/lite/examples/object_detection/raspberry_pi/detect_picamera.py
#
# bash download.sh /tmp
#
# Install Kasa: pip3 install python-kasa
# Add to path: export PATH="/home/pi/.local/bin:$PATH"
#
# Example
# python3 dog_fan.py  --model /tmp/detect.tflite --labels /tmp/coco_labels.txt 
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import re
import time

import numpy as np
import picamera

from PIL import Image
from tflite_runtime.interpreter import Interpreter

from gpiozero import CPUTemperature

# Kasa stuff
import asyncio
from kasa import SmartPlug

def load_labels(path):
    """Loads the labels file. Supports files with or without index numbers."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = {}
        for row_number, content in enumerate(lines):
            pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()
    return labels

def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image

def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor

def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all output details
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
  return results

def main():
    print("Started Program")
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', help='Path to .tflite file', required=True)

    parser.add_argument('--labels', help='Path to labels file', required=True)

    parser.add_argument('--threshold', help='Detection threshold', required=False,
            type=float, default=0.5)

    parser.add_argument('--cam_width', help='Camera width', required=False,
            type=int, default=640)

    parser.add_argument('--cam_height', help='Camera height', required=False,
            type=int, default=480)

    args = parser.parse_args()

    labels = load_labels(args.labels)
    interpreter = Interpreter(args.model)
    interpreter.allocate_tensors()

    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']    

    # declare smartplug to control the fan
    plug = SmartPlug("192.168.0.29")

    print("[main] Starting camera")
    with picamera.PiCamera(resolution=(args.cam_width, args.cam_height)) as camera:
       
        print("[main] Starting main loop\n")
        # camera.start_preview()     # Left in for debugging

        try: 
            stream = io.BytesIO()
            remaining_fan_time = time.monotonic()
            last_loop_time = time.monotonic()

            while 1:
                loop_time = time.monotonic() - last_loop_time
                last_loop_time = time.monotonic()


                cpu_temp = CPUTemperature()

                if cpu_temp.temperature < 80 :
                    
                    camera.capture(stream, format='jpeg', use_video_port=True)

                    stream.seek(0)

                    image = Image.open(stream).convert('RGB').resize(
                                                (input_width, input_height), 
                                                Image.ANTIALIAS)
                    
                    results = detect_objects(interpreter, image, args.threshold)

                    # Print program statistics
                    print("\n----------------")
                    print("Dog Fan Project")
                    print("----------------")

                    print("Loop Time:", round(loop_time, 2), "FPS:", round(1/loop_time, 2))

                    # Print all detected objects
                    print("\nDetected objects")
                    for obj in results:
                        print(labels[obj['class_id']], round(obj['score'], 2))

                    # Checking for objects of interest
                    print("\nObjects of Interest")
                    for obj in results:
                        # if (labels[obj['class_id']] == "person") or (labels[obj['class_id']] == "dog") :
                        if (labels[obj['class_id']] == "dog")  or (labels[obj['class_id']] == "cat"):
                            print("FOUND", labels[obj['class_id']], "-", round(obj['score'], 2))
                            remaining_fan_time = time.monotonic() + 5
                            break

                    if remaining_fan_time > time.monotonic() :
                        print("\nFan remaining time:", round(remaining_fan_time - time.monotonic(), 1))
                        asyncio.run(plug.turn_on())
                    else :
                        print("\nFan remaining time: OFF")
                        asyncio.run(plug.turn_off())

                    stream.seek(0)
                    stream.truncate()

                else :
                    print("ERROR Overtemp! Not running detector: ", round(cpu_temp.temperature, 1))            
                    time.sleep(1)

        finally:
            # camera.stop_preview()     # Left in for debugging
            pass

    

if __name__ == '__main__':
    main()
