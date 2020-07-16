import cv2
import os
import sys
import logging as log
import numpy as np
from openvino.inference_engine import IENetwork, IECore

class gaze_estimation:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        Set your instance variables.
        '''
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self):
        '''
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.plugin = IECore()
        model_xml = self.model_name
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
       
        self.network = IENetwork(model=model_xml, weights=model_bin)

        self.check_model()
        self.exec_network = self.plugin.load_network(self.network, self.device)
        
        self.output_blob = next(iter(self.network.outputs))
        return

    def predict(self,net_input,request_id):
        '''
        This method is meant for running predictions on the input image.
        '''
        self.exec_network.start_async(request_id=request_id,inputs=net_input)
        return

    def get_output(self):
        return self.exec_network.requests[0].outputs[self.output_blob]

    def check_model(self):
        if self.extensions and "CPU" in self.device:
            self.plugin.add_extension(self.extensions, self.device)

        supported_layers = self.plugin.query_network(network=self.network, device_name="CPU")    
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)

    def wait(self):
        status = self.exec_network.requests[0].wait(-1)
        return status

    def preprocess_input(self, left_eye, right_eye, head_pose_angle):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        h = self.network.inputs['left_eye_image'].shape[2]
        w = self.network.inputs['left_eye_image'].shape[3]
        p_frame_l = cv2.resize(left_eye, (w, h))
        p_frame_l = p_frame_l.transpose((2,0,1))
        p_frame_l = p_frame_l.reshape(1, 3, h, w)

        h = self.network.inputs['right_eye_image'].shape[2]
        w = self.network.inputs['right_eye_image'].shape[3]
        p_frame_r = cv2.resize(right_eye, (w, h))
        p_frame_r = p_frame_r.transpose((2,0,1))
        p_frame_r = p_frame_r.reshape(1, 3, h, w)
        
        input_names = ['left_eye_image','right_eye_image','head_pose_angles']
        net_input = {input_names[0]:p_frame_l,input_names[1]:p_frame_r,input_names[2]:head_pose_angle}
        return net_input

    def preprocess_output(self, outputs,left_eye_center,right_eye_center,face_frame,args):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        x,y = outputs[0][0],outputs[0][1]
        if args.log_level == "DEBUG":
            cv2.arrowedLine(face_frame, left_eye_center, (int(left_eye_center[0]+x*400), int(left_eye_center[1]-y*400)), (0,0,0), 5)
            cv2.arrowedLine(face_frame, right_eye_center, (int(right_eye_center[0]+x*400), int(right_eye_center[1]-y*400)), (0,0,0), 5)
        return x,y
