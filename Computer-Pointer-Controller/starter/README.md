# Computer Pointer Controller

In this project, I use a gaze detection model to control the mouse pointer of my computer. I use the Gaze Estimation model to estimate the gaze of the user's eyes and change the mouse pointer position accordingly. This project will require running multiple models in the same machine and coordinating the flow of data between those models.

![Output sample](https://github.com/mkhoshle/IntelEdgeAI-Projects/blob/master/Computer-Pointer-Controller/starter/output_video.gif)

## Project Set Up and Installation
The models used in this project are:

- [Face Detection](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
- [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
- [Facial Landmark Detection](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
- [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

The procedure for seting up your environment includes:
- Download and install **OpenVino toolkit** using the instructions provided [here](https://github.com/udacity/nd131-openvino-fundamentals-project-starter/blob/master/mac-setup.md).
- Download and install the **DL Workbench** from [here](https://docs.openvinotoolkit.org/latest/_docs_Workbench_DG_Install_Workbench.html). In short, you will need to:
	- For Linux and Windows you can find the docker command [here](https://docs.openvinotoolkit.org/latest/_docs_Workbench_DG_Install_from_Docker_Hub.html#install_dl_workbench_from_docker_hub_on_windows_os).
	- For macOS, the Docker command is:
	```
	docker run -p 127.0.0.1:5665:5665 --name workbench -e PROXY_HOST_ADDRESS=0.0.0.0 -e PORT=5665 -it openvino/workbench:latest
	```
- Download **VTune Amplifier** from [here](https://software.intel.com/en-us/vtune/choose-download#standalone) and you can get detailed instructions about how to install and run it from [here](https://software.intel.com/en-us/get-started-with-vtune). 

This code follows the following directory structure:

	.
	├── LICENSE 
	├── README.md                      <- The top-level README for developers using this project.
	├── bin
	│   ├── demo.mp4                   <- Input video to test the code.
	│     
	│     
	├── requirements.txt               <- Package dependencies required to be installed.
	├── src                            <- Source code for use in this project.
        ├── face_detection.py                  <- Face detection model
        ├── head_pose_estimation.py            <- Head pose estimation model
        ├── facial_landmarks_detection.py      <- Facial landmarks detection model
        ├── gaze_estimation.py                 <- Gaze estimation model
        │
        ├── input_feeder.py                    <- Script to handle the input images
        ├── mouse_controller.py                <- Script to move the mouse accoording to estimated gaze direction
        │
        └── main.py                            <- The main script coordinating flow of data from the input, and then amongst different models and finally to the mouse controller.
    
## Model Pipeline:
The project involves interaction multiple models and here is the model pipeline works:
- Face detection model receives an image, run the inference, detects the face, crop and output the detected face.
- Head pose estimation model receives the dtected face as input and output the head pose angles (yaw, pitch, roll).
- Landmark face detection model receives the dtected face as input and output the cropped left and right eye. 
- Gaze estimation model receives the cropped left and right eye and the head pose angles as input and output the gaze_vector.
- Finally the x and y coordinates of the gaze vector will be provided to the mouse controller and the mouse pointer will be moved accordingly. 

## Benchmarks
The benchamrk results are shown in the following tables. These throughputs and latencies are calculated using DL Workbench for various model precisions. I used my laptop with `Intel(R) Core(TM) i5-5257U CPU @ 2.70GHz` to perform these benchmarks. 

**Note:** For benchamark and performance optimization purpose, these lines of the code should be commented out.
```
cv2.imshow('window-name',frame)
# This will cause a delay in order to give mouse enough time to move in the desired direction
cv2.waitKey(60)
mc.move(x,y)
```

<img src="benchmak.png"/>

To run the model you can use the following sample:
1) For base-case:
```
python main.py -i ../bin/demo.mp4  -m "../intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml ../intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml ../intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml ../intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml" -d CPU -it 'video' -pt 0.7 -cp True > profile_output.txt
```
2) For the second case:
```
python main.py -i ../bin/demo.mp4  -m "../intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml ../intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml ../intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml  ../intel/gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002.xml" -d CPU -it 'video' -pt 0.7 -cp True > profile_output.txt
```

## Results
As highlighted in the table, Face detection model with precision of FP32, Head pose estimation with FP16, Facial landmark detection with FP16 and gaze estimation with FP16-INT8 precisions provide higher throughput and lower latencies as compared to other precisions of the model.  
In the base-case all models have FP32 precision. Considering the throughput and latency results, the second case includes Face detection model with precision of FP32, Head pose estimation with FP16, Facial landmark detection with FP16 and gaze estimation with FP16-INT8 which has improved the overal timing a little bit. 

Note that, changing the precision of Head pose estimation, Facial landmark detection, and gaze estimation had minor improvemnt on the overal execution time. The reason is because the Face detection model is the main bottleneck as shown on the profiling output from the [base case](https://github.com/mkhoshle/IntelEdgeAI-Projects/blob/master/Computer-Pointer-Controller/starter/profile_output-BaseCase.txt) and the [second case](https://github.com/mkhoshle/IntelEdgeAI-Projects/blob/master/Computer-Pointer-Controller/starter/profile_output-SecondCase.txt) as well.

## Demo
To run the model you can use the following sample:
1) For debugging purpose and to get all the bounding boxs and stats on the image:
```
python main.py -i ../bin/demo.mp4  -m "../intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml ../intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml ../intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml ../intel/gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002.xml" -d CPU -it 'video' -pt 0.7 -ll "DEBUG"
```

2) For production purpose:
```
python main.py -i ../bin/demo.mp4  -m "../intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml ../intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml ../intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml ../intel/gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002.xml" -d CPU -it 'video' -pt 0.7
```

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
