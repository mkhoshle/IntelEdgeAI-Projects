import os
import sys
import time
import cv2
import numpy as np
import logging as log

from argparse import ArgumentParser
from face_detection import face_detection
from gaze_estimation import gaze_estimation
from head_pose_estimation import head_pose_estimation
from facial_landmarks_detection import facial_landmarks_detection
from mouse_controller import MouseController
from input_feeder import InputFeeder

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="List of paths to an xml file with a trained model.")
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
                        help="The type of input can be 'video','image' or 'cam'")
    parser.add_argument("-it", "--input_type", type=str, default='video',
                        help="The type of input can be 'video','image' or 'cam'")
    return parser


def infer_on_stream(args,client):
    feed = InputFeeder(input_type=args.input_type,input_file=args.input)
    feed.load_data()

    width = int(feed.cap.get(3))
    height = int(feed.cap.get(4))
   
    model1, model2, model3,model4 = list(args.model.split(' '))
    # Load face dtection model
    plugin_fd = face_detection(model1, args.device)
    plugin_fd.load_model()
    
    # Load head pose estimation model
    plugin_hp = head_pose_estimation(model2, args.device)
    plugin_hp.load_model()

    # Load landmark dtection model
    plugin_fld = facial_landmarks_detection(model3, args.device)
    plugin_fld.load_model()

    # Load gaze estimation model
    plugin_ge = gaze_estimation(model4, args.device)
    plugin_ge.load_model()

    # Instansiate Mouse Controller
    mc = MouseController('high','fast')

    request_id = 0  
    for i,frame in enumerate(feed.next_batch()):
        if frame is None:
            break
        else:
            print("iteration No: {}".format(i))
            key_pressed = cv2.waitKey(60)
        
            ## Face Detection
            p_frame = plugin_fd.preprocess_input(frame)
            plugin_fd.predict(p_frame, request_id)
            if plugin_fd.wait() == 0:
                outputs = plugin_fd.get_output()
                face_crop = plugin_fd.preprocess_output(outputs, frame, args, width, height)

            ## Head pose estimation
            p_face_crop = plugin_hp.preprocess_input(face_crop)
            plugin_hp.predict(p_face_crop, request_id)
            if plugin_hp.wait() == 0:
                outputs = plugin_hp.get_output()
                angles = plugin_hp.preprocess_output(outputs)

            ## Landmark detection model
            p_face_crop = plugin_fld.preprocess_input(face_crop)
            plugin_fld.predict(p_face_crop, request_id)
            if plugin_fld.wait() == 0:
                outputs = plugin_fld.get_output()
                initial_w = np.shape(face_crop)[1]
                initial_h = np.shape(face_crop)[0]
                left_eye,right_eye = plugin_fld.preprocess_output(outputs,face_crop,initial_w, initial_h)
        
            ## Gaze estimation model
            net_input = plugin_ge.preprocess_input(left_eye, right_eye,angles)
            plugin_ge.predict(net_input, request_id)
            if plugin_ge.wait() == 0:
                outputs = plugin_ge.get_output()
                x,y = plugin_ge.preprocess_output(outputs)
            
            if key_pressed == 27:
                break

            ## Change mouse location
            cv2.imshow('window-name',frame)
            mc.move(x,y)
            # This will cause a delay in order to give mouse enough time to move in the desired direction
            cv2.waitKey(30)

    feed.close()
    return

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Perform inference on the input stream
    start_time = time.time()
    infer_on_stream(args,client)
    print("Inference time was: {}".format(time.time()-start_time))


if __name__ == '__main__':
    main()
