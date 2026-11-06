"""Recognize objects in a picture or a live USB camera feed."""

import argparse
import base64
from pathlib import Path

import cv2

from camera_vision import CameraVision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', type=Path)
    source.add_argument('--camera', type=int, help='USB camera index')
    parser.add_argument('--targets', default='', help='Comma-separated objects to look for')
    parser.add_argument('--describe', action='store_true', help='Summarize the detected objects')
    parser.add_argument('--output', type=Path, help='Save an annotated JPEG in image mode')
    args = parser.parse_args()
    if args.output is not None and args.image is None:
        parser.error('--output requires --image')

    vision = CameraVision()
    if args.image is not None:
        jpeg = args.image.read_bytes()
        result = vision.analyze(jpeg, args.targets, args.describe)
        for item in result['boxes']:
            print(f"{item['label']}: {item['confidence']:.0%}")
        if args.describe:
            print(result['description'])
        if not result['boxes'] and not args.describe:
            print('No matching objects detected.')
        if args.output is not None:
            args.output.write_bytes(base64.b64decode(result['annotated']) if result['annotated']
                                    else jpeg)
        return

    camera = cv2.VideoCapture(args.camera)
    previous_description = ''
    try:
        if not camera.isOpened():
            parser.exit(1, f'Cannot open camera {args.camera}\n')
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        camera.set(cv2.CAP_PROP_FPS, 10)
        while True:
            ok, frame = camera.read()
            if not ok:
                parser.exit(1, 'Camera stopped sending frames.\n')
            ok, jpeg = cv2.imencode('.jpg', frame)
            if not ok:
                parser.exit(1, 'Could not encode camera frame.\n')
            result = vision.analyze(jpeg.tobytes(), args.targets, args.describe)
            if args.describe and result['description'] != previous_description:
                previous_description = result['description']
                print(previous_description, flush=True)
            if result['annotated']:
                import numpy as np
                data = np.frombuffer(base64.b64decode(result['annotated']), np.uint8)
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            cv2.imshow('RoboRecognizer (Q to quit)', frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
