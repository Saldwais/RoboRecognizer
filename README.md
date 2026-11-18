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

## Overhead sock sorting

Use a fixed camera, consistent lighting and an empty reference photo of the taped table. Spread socks out flat in the pickup area and keep the arm out of view. Start each batch with empty boxes. The background detector treats separate foreground objects in that area as sock candidates.

Save a layout JSON with rectangles in overhead image pixels, using `[x, y, width, height]`. These sample coordinates illustrate the format for a 640 × 480 image. Measure your own taped areas and keep the boxes in the same order between runs.

```json
{"pickup":[150,30,300,420],"left":[10,30,130,420],"boxes":[[460,15,165,100],[460,125,165,100],[460,235,165,100],[460,345,165,100]]}
```

```powershell
python sock_sort.py --camera 1 --empty empty-table.png --layout layout.json --annotated reviewed.jpg --review > plan.json
```

Use `--image loaded-table.png` for a saved frame. Use `--flip-horizontal` if the physical right is on the left side of the image, and measure all layout coordinates in the flipped view. Compare the crops in the review window. Press **M** only if the full pattern, all logos, printed words and pictures agree. Press **D** when you can establish a difference. Press **U** if a detail could be hidden or if the sock looks folded. Spread or turn the held socks so their details are visible and capture a fresh plan before moving anything. The visible difference percentage is a pixel comparison to help review. It has no calibrated meaning as a probability. The size and color thresholds require checking on the actual table.

Every pair in a group must be approved. A group of any size above one gets one box. Confirmed singles go left. Conflicting matches, socks requiring reinspection and groups beyond the four available boxes are held. Without `--review`, plausible matches are held for review. The JSON contains image coordinates and assignments for a downstream controller.

### SO-101 recordings and ACT

Use the recording environment or a dedicated Python environment with `lerobot[training]==0.6.1`, PyAV and PyArrow. ACT means **Action Chunking Transformer**. It learns a sequence of joint commands from the recorded camera frames, follower state and leader demonstrations. Training and evaluation run offline and need no robot connection.

```powershell
python sock_recordings.py PATH_TO_DATASET --previews audit-previews
python sock_recordings.py PATH_TO_DATASET --train act-socks --holdout 10 11 --device cuda
python sock_recordings.py PATH_TO_DATASET --evaluate PATH_TO_PRETRAINED_MODEL --holdout 10 11 --device cuda
```

The audit checks every joint data row and samples the beginning, middle and end of each episode for each camera. Preview files help identify the camera angles and inspect the demonstrated behavior. Training uses audited episodes and leaves out the specified episode IDs. For a CPU smoke test use `--device cpu --steps 1 --batch-size 1 --from-scratch`. Point evaluation at a numbered checkpoint's `pretrained_model` directory. Evaluation measures first-action error against recorded movements and against a baseline that holds the current joint positions. Physical pick and place success needs separate trials.

On the existing two-camera recordings, `camera0` is the gripper view and `camera1` is the elevated table view. The dataset label describes only left placement. Keep the same input views and follower calibration when eventually testing the left policy on COM5. COM3 is the leader used for demonstrations. To drive the right boxes from the planner, collect demonstrations that specify both the intended sock and destination, or train separate destination policies with an explicit way to identify the intended sock. A single left-placement checkpoint has no interface for selecting a sock in a mixed pile or selecting a right box.

## RoboPilot integration

RoboRecognizer can be used on its own. In the local [RoboPilot](https://github.com/Saldwais/robopilot) integration, all three XLeRobot cameras have independent recognition settings. The center camera starts with detection enabled. Each wrist camera can enable it when needed. Each view has a custom object field and a Describe view button.

## Roadmap

I’ll keep developing RoboRecognizer and testing it with the XLeRobot center and wrist cameras. The next steps are more reliable recognition of everyday objects, steadier detection across moving frames, and a verified ROS 2 camera and detection demo. Later, I plan to add LiDAR to combine camera labels with measured distances and improve the robot’s view of nearby obstacles.

RoboRecognizer uses [Ultralytics YOLO and YOLO-World](https://docs.ultralytics.com/models/yolo-world). The project code is MIT licensed. The upstream models and dependencies have their own license terms.
