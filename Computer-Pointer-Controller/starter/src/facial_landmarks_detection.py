import cv2
import os
import sys
import logging as log
import numpy as np
from openvino.inference_engine import IENetwork, IECore

class facial_landmarks_detection:
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

        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        return

    def predict(self, image, request_id):
        '''
        This method is meant for running predictions on the input image.
        '''
        self.exec_network.start_async(request_id=request_id,inputs={self.input_blob: image})
        return

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

    def get_output(self):
        return self.exec_network.requests[0].outputs[self.output_blob]

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        net_input_shape = self.network.inputs[self.input_blob].shape
        h = net_input_shape[2]
        w = net_input_shape[3]
        p_frame = cv2.resize(image, (w, h))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, 3, h, w)
        return p_frame

    def preprocess_output(self, outputs, image, initial_w, initial_h, args):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        outputs= outputs[0]
        xl,yl = int(outputs[0][0]*initial_w),int(outputs[1][0]*initial_h)
        xr,yr = int(outputs[2][0]*initial_w),int(outputs[3][0]*initial_h)
        
        # make box for left eye
        xlmin = xl-15
        ylmin = yl-15
        xlmax = xl+15
        ylmax = yl+15
        if args.log_level == "DEBUG":
            cv2.rectangle(image, (xlmin, ylmin), (xlmax, ylmax), (0, 255, 0), 3)        
        left_eye = image[ylmin:ylmax, xlmin:xlmax]
        left_eye_center = (xl,yl)

        # make box for right eye
        xrmin = xr-15
        yrmin = yr-15
        xrmax = xr+15
        yrmax = yr+15
        if args.log_level == "DEBUG":
            cv2.rectangle(image, (xrmin, yrmin), (xrmax, yrmax), (0, 255, 0), 3)
        right_eye = image[yrmin:yrmax, xrmin:xrmax]
        right_eye_center = (xr,yr)

        return left_eye,right_eye,left_eye_center,right_eye_center
