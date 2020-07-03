# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required questions.

## Explaining Custom Layers
 
**Note**: The DSOD model I used did not have any custom layer.

The process behind converting custom layers involves 1) registering the custom layers as extensions to the Model Optimizer. For Caffe, the second option is to register the layers as Custom, then use Caffe to calculate the output shape of the layer which requires Caffe on your system. For TensorFlow, its second option is to actually replace the unsupported subgraph with a different subgraph. 3) The final TensorFlow option is to perform the computation of the subgraph in TensorFlow during inference.

Some of the potential reasons for handling custom layers are that not all layers used in Neural Networks are among standard known layers available and supported in various frameworks. In fact, OpenVino has list of supported layers and the layers that are not in that list need to be converted.

## Comparing Model Performance

Comparison of the models before and after conversion to Intermediate Representations:

**Model Accuracy:** The model accuracy pre-conversion is usuallay a little higher than post-conversion. The reason is that the optimization of the modle usually leads to the drop in the accuracy.

THIS RESULTS ARE FOR SSD_MOBILENET
In the video the people come into the scene and leave at following frame NO:
    in - out
p1: 61 - 198 
p2: 228 - 449
p3: 503 - 700
p4: 746 - 869
p5: 922 - 1200
p6: 1236 - 1360
Therefore, total No of frames at which a person exists is: 1080

Running the model **post-conversion**, I got the following result when a person come into the scene and leave:
    in - out
p1: 70 - 72
p2: 236 - 241
p3: 512 - 516
p4: 759 - 780
p5: 1182 - 1183
p6: 1243 - 1251

As can be seen all the frames at which it is predicted that a person enter and leave are all whitin the period in the originial video shown above. Because the probability thresold I used is pretty high it makes sense to me that the number of frame the model predicts that the person is in the scene is much lower than what it actually is in the original video. 
Total No of frames in original video: 1394
True Positives: 2,5,4,91,1,8 = 111
False Negatives: 135,216,193,32,277,116 = 969
True Negative: 1394-1080 =314
Accuracy = (TP+TN)/Total = (314+111)/1394 = 0.3

_Note:_ I do not have the model accuracy for the pre-converted model. I have created the code for running inference on the pre-converted model (Inference-MobileNet-Preconverted.ipynb) but on laptop was too slow. So I could not run it. 

**Model Size:** I checked model size using ls -sh command. The size of the model pre- and post-conversion was 57MB and 57MB (for DSOD) respectively and 67MB and 65MB (for SSD-Mobilenet-v2) respectively. Overal pre-converted is a little higher in size than post-converted model.

**Inference Time:** The inference time of the model pre-conversion was taking hours. I tested the preconversion model inference time for SSD-Mobilenet-v2 on my laptop and it was very slow. I have uploaded my code (Inference-MobileNet-Preconverted.ipynb) for running the inference on the pre-converted model(.pb file). Since it was too slow I did not run the inference for all frames. Overal there are 1394 frames that I obtained using `int(capture.get(cv2.CAP_PROP_FRAME_COUNT))`. The inference time for a single frame was 11.31s. Because all frames are of the same size, we can say that the total inference time will be about 15766.14s approximately. These timings are for SSD-Mobilenet-v2 model. And model post-conversion was 438.8s(SSD-Mobilenet-v2), 1035.56s(DSOD) respectively. The difference between 15766.14 and 438.8s is huge.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are to measure the number of people that pass through a certain passage or entrance. The information people counters provide is used at all levels of businesses, from planning front-line activities to setting overall strategy. Besides, it can be used for analyzing store performance or crowd statistics during festivals. 

Each of these use cases would be useful because once collected, count data is normally sent via the internet (in near real-time) to a retail analytics platform for analysis. The data can be analyzed and used for further decision making and planning to improve different aspects of the business.


## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows:

**camera focal length:** All lenses suffer from a certain amount of distortion, in which the image is either stretched or compressed in a non-linear way, making accurate measurements difficult. In general, shorter focal length lenses experience more distortion than longer focal length lenses since the light hits the sensor from a bigger angle.

**Lighting:** Non-uniform illumination and shading across the image affect lens performance. With any lens, image brightness is reduced towards the edges and this is known as vignetting. Cos4 vignetting occurs because the light has to travel further to the edge of the image and reaches the sensor at a shallower angle. Mechanical vignetting occurs when the light beam is mechanically blocked, usually by the lens amount.

**Model Accuracy:** Accuracy of the deployed edge model is very important. The more accurate the model the lower the threshold needed to be provided to the app will be. 

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three different models. However, you may also use this heading to detail how you converted a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [LSCCNN]
  - [https://github.com/val-iisc/lsc-cnn/tree/master/utils]
  - I converted the model to an Intermediate Representation with the code named Convert-lsccnn-to-IR.ipynb. 
  - The model was insufficient for the app because the model was too complicated and heavy for edge application. The size was an issue to load in buffer. If we want to use such a model for edge application, it would require model optimization techniques. This is because on edge there will not be enough memory resources for such complicated models. The error I got was:
  ```Traceback (most recent call last):
  File "main.py", line 219, in <module>
    main()
  File "main.py", line 215, in main
    infer_on_stream(args, client)
  File "main.py", line 109, in infer_on_stream
    plugin.load_model(args.model, args.device, args.cpu_extension)
  File "/home/workspace/inference.py", line 54, in load_model
    self.network = IENetwork(model=model_xml, weights=model_bin)        
  File "ie_api.pyx", line 415, in openvino.inference_engine.ie_api.IENetwork.__cinit__
   RuntimeError: segment exceeds given buffer limits. Please, validate weights file.
   ```

  - I did not try to improve the model for the app and decided to look for smaller, more efficient models.
  
- Model 2: [Efficientdet]
  - [https://github.com/google/automl/tree/master/efficientdet]
  - I converted the model to an Intermediate Representation with the following arguments:
  ```!rm  -rf savedmodeldir
  !python model_inspect.py --runmode=saved_model --model_name=efficientdet-d0 \
   --ckpt_path=efficientdet-d0 --saved_model_dir=savedmodeldir \
   --tensorrt=FP32 --tflite_path=efficientdet-d0.tflite.```
   
 And then, 
  
  ```python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model efficientdet-   d0_frozen.pb --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json```
    
But I got the following error:
  ```np_resource = np.dtype([("resource", np.ubyte, 1)])
  [ ERROR ]  Failed to match nodes from custom replacement description with id                     'ObjectDetectionAPIPreprocessorReplacement':
  It means model and custom replacement description are incompatible.
  Try to correct custom replacement description according to documentation with respect to model   node names
  [ ERROR ]  Failed to match nodes from custom replacement description with id                     'ObjectDetectionAPISSDPostprocessorReplacement':
  It means model and custom replacement description are incompatible.
  Try to correct custom replacement description according to documentation with respect to model   node names
  [ ERROR ]  Cannot infer shapes or values for node "TensorArrayV2".
  [ ERROR ]  Tensorflow type 21 not convertible to numpy dtype.
  [ ERROR ]  
  [ ERROR ]  It can happen due to bug in custom shape infer function <function                     tf_native_tf_node_infer at 0x7fa816fb57b8>.
  [ ERROR ]  Or because the node inputs have incorrect values/shapes.
  [ ERROR ]  Or because input shapes are incorrect (embedded to the model or passed via           --input_shape).
  [ ERROR ]  Run Model Optimizer with --log_level=DEBUG for more information.
  [ ERROR ]  Tensorflow type 21 not convertible to numpy dtype.
  Stopped shape/value propagation at "TensorArrayV2" node. 
  For more information please refer to Model Optimizer FAQ (https://docs.openvinotoolkit.org       /latest/_docs_MO_DG_prepare_model_Model_Optimizer_FAQ.html), question #38. 
  Exception occurred during running replacer "REPLACEMENT_ID" (<class                             'extensions.middle.PartialInfer.PartialInfer'>): Tensorflow type 21 not convertible to numpy     dtype.
  Stopped shape/value propagation at "TensorArrayV2" node. 
  For more information please refer to Model Optimizer FAQ (https://docs.openvinotoolkit.org       /latest/_docs_MO_DG_prepare_model_Model_Optimizer_FAQ.html), question #38
  ```
  
  - The model was insufficient for the app because Efficientdet is a new network and probably written in Tensorflow 2. Therefore my guess is that this network is not supported by OpenVino yet, efficientdet is not found in https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html.
  - I tried to look for other models.

- Model 3: [ssd_mobilenet_v2_coco_2018_03_29]
  - [OpenVino Model ZOO]
  - I converted the model to an Intermediate Representation with the following arguments:
  ```python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config     --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json```
  
  - The model was sufficient for the app. Because this model was a pretty small and efficiet model, I used it for debugging purpose and it was very helpful for me to get my code working.

  - **Note**: To run the model you should use the following command:
  ```python main.py -i resources/Pedestrian_Detect_2_1_1.mp4  -m model3-SSD-MobileNet/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.9```

- Model 4: [DSOD]
  - [https://github.com/szq0214/DSOD]. Here is the link to the model I downloaded for the project: https://drive.google.com/drive/folders/0B4cvsEOB5eUCaGU3MkRkOENRWWc. The model is: DSOD300 (07+12) bs=4.
  - I converted the model to an Intermediate Representation with the following arguments:
  ```python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model DSOD300_VOC0712.caffemodel --input_proto deploy.prototxt```
  
  - The model was sufficient for the app and I successfully converted it to IR representation.
  - This model is my final model I used for answering the project's questions.
  
  - **Note**: To run the model you should use the following command:```python main.py -i resources/Pedestrian_Detect_2_1_1.mp4  -m model4-DSOD/DSOD300_VOC0712.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.85```
