"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np
import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    ''' 
    center_x = None 
    center_y = None 
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= args.prob_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            center_x = (xmin+xmax)/2
            center_y = (ymin+ymax)/2
    
    return frame, center_x, center_y


def infer_on_stream(args,client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    plugin = Network()
    
    args.prob_threshold = float(args.prob_threshold)

    ### TODO: Load the model through `infer_network` ###
    plugin.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = plugin.get_input_shape()
    
    ### TODO: Handle the input stream ###
    capture = cv2.VideoCapture(args.input)
    capture.open(args.input)
    
    image_flag = False
    if args.input=='cam':
        args.input = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') or args.input.endswith('.png'):
        image_flag = True
    
    # Grab the shape of the input 
    width = int(capture.get(3))
    height = int(capture.get(4))   
    
    if not image_flag:
        out = cv2.VideoWriter('out.mp4', 0x00000021, 24, (width,height))
    else:
        out = None
        
    ### TODO: Loop until stream is over ###
    current_count = 0
    last_count = 0 
    total_count = 0
    center_x_old = 0            # x component of the box center
    center_y_old = 0            # y component of the box center
    request_id = 0   
    ii = 0                      # Frame number
    while capture.isOpened():
        ii += 1
        ### TODO: Read from the video capture ###
        flag, frame = capture.read()
        if not flag:
            break
        
        key_pressed = cv2.waitKey(60)
        
        ### TODO: Pre-process the image as needed ###
        h = net_input_shape[2]
        w = net_input_shape[3]
        p_frame = cv2.resize(frame, (w, h))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, 3, h, w)
        
        ### TODO: Start asynchronous inference for specified request ###
        plugin.exec_net(p_frame,request_id)
        
        ### TODO: Get the results of the inference request ###
        ### TODO: Wait for the result ###
        if plugin.wait() == 0:
            ### TODO: Extract any desired stats from the results ###
            result = plugin.get_output()
            
            ### Update the frame to include detected bounding boxes
            out_frame, center_x, center_y = draw_boxes(frame, result, args, width, height)  

            if center_x:
                if np.sqrt((center_x-center_x_old)**2+(center_y-center_y_old)**2)>110:
                    current_count = 1
                center_x_old = center_x
                center_y_old = center_y
            else:
                current_count = 0
                
            client.publish("person", json.dumps({"count": current_count}))
            ### Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            if current_count > last_count:
                start_time = time.time()
                total_count = total_count + current_count - last_count
                client.publish("person", json.dumps({"total": total_count}))
#                 print ('frame in', ii)     # This is for debugging 

            if current_count < last_count:
                duration = int(time.time() - start_time)
                # Publish messages to the MQTT server
                client.publish("person/duration",json.dumps({"duration": duration}))         
#                 print ('frame out', ii)    # This is for debugging 
                
                    
            if key_pressed == 27:
                break
            last_count = current_count    
              
        out.write(out_frame)
        sys.stdout.buffer.write(out_frame)  
        sys.stdout.flush()            
         
    
    if not image_flag:
        out.release()
        
    capture.release()
    cv2.destroyAllWindows()
    ### TODO: Disconnect from MQTT
    client.disconnect()
    

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    start_time = time.time()
    infer_on_stream(args,client)
    print("Inference time was: {}".format(time.time()-start_time))


if __name__ == '__main__':
    main()
