# RoboRecognizer by Saleh Aldwais

RoboRecognizer helps your robot camera understand what's in view. It draws boxes around objects, lets you type what you want it to look for, and can describe the view in plain language using the objects it detects. It started as a camera experiment on XLeRobot and also works with a standard USB camera or a saved image.

## What works today

- Recognize common objects such as apples, bananas, bottles, chairs, and refrigerators.
- Look for custom targets such as a recycling bin or a soda can by entering their names.
- Show labels and confidence scores and summarize the objects in the frame.
- Run the detector locally. The first run downloads model weights. Custom targets also use the YOLO-World text model.

Descriptions summarize detections. They can miss objects or give an incorrect label. Brand names such as Diet Coke need testing on the actual packaging.

## Run it

Use Python 3.10 or newer in a virtual environment and install the dependencies from this directory. Git is required to install the text model dependency.

```sh
python -m pip install -r requirements.txt
python recognize.py --camera 0 --describe
```

Try a different index if you have more than one camera. Press **Q** to close the preview. To test without a live camera, use your own JPEG image:

```sh
python recognize.py --image kitchen.jpg --targets "banana, apple, refrigerator" --describe --output labeled.jpg
```

## RoboPilot integration

RoboRecognizer can be used on its own. In the local [RoboPilot](https://github.com/Saldwais/robopilot) integration, all three XLeRobot cameras have independent recognition settings. The center camera starts with detection enabled. Each wrist camera can enable it when needed. Each view has a custom object field and a Describe view button.

## Roadmap

I’ll keep developing RoboRecognizer and testing it with the XLeRobot center and wrist cameras. The next steps are more reliable recognition of everyday objects, steadier detection across moving frames, and a verified ROS 2 camera and detection demo. Later, I plan to add LiDAR to combine camera labels with measured distances and improve the robot’s view of nearby obstacles.

RoboRecognizer uses [Ultralytics YOLO and YOLO-World](https://docs.ultralytics.com/models/yolo-world). The project code is MIT licensed. The upstream models and dependencies have their own license terms.
