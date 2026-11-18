"""Plan sock sorting from a fixed overhead camera and its empty reference frame."""
import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np


def detect(frame, empty, pickup):
    x, y, w, h = pickup
    delta = cv2.absdiff(frame, empty)[y:y+h, x:x+w].max(axis=2)
    mask = np.uint8(delta > 26) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    socks = []
    for contour in sorted(contours, key=lambda c: cv2.boundingRect(c)[:2]):
        if cv2.contourArea(contour) < 250:
            continue
        contour = contour + np.array([x, y], dtype=np.int32)
        bx, by, bw, bh = cv2.boundingRect(contour)
        silhouette = np.zeros(frame.shape[:2], np.uint8)
        cv2.drawContours(silhouette, [contour], -1, 255, cv2.FILLED)
        pixels = cv2.cvtColor(frame[by:by+bh, bx:bx+bw], cv2.COLOR_BGR2LAB)
        pixels = pixels[silhouette[by:by+bh, bx:bx+bw] != 0]
        size = sorted(cv2.minAreaRect(contour)[1])
        clear = bx > x and by > y and bx+bw < x+w and by+bh < y+h
        crop = frame[by:by+bh, bx:bx+bw]
        inside = silhouette[by:by+bh, bx:bx+bw]
        if bw > bh:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            inside = cv2.rotate(inside, cv2.ROTATE_90_CLOCKWISE)
        print_view = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2LAB), (48, 96))
        inside = cv2.resize(inside, (48, 96), interpolation=cv2.INTER_NEAREST).astype(bool)
        socks.append({
            'id': len(socks) + 1, 'bbox': [bx, by, bw, bh],
            'size': size, 'color': np.median(pixels, axis=0),
            'clear': clear, 'print_view': print_view, 'inside': inside,
        })
    return socks


def classify(a, b):
    if not (a['clear'] and b['clear']):
        return 'review'
    ratio = min(min(x, y) / max(x, y) for x, y in zip(a['size'], b['size']))
    color = np.linalg.norm(a['color'] - b['color'])
    if ratio < .60 or color > 65:
        return 'different'
    if ratio < .85 or color > 25:
        return 'review'
    return 'inspect_print'


def print_difference(a, b):
    """Rank visible print discrepancies, keeping the decision in the review step."""
    scores = []
    for rotation in (slice(None), slice(None, None, -1)):
        overlap = a['inside'] & b['inside'][rotation, rotation]
        if overlap.sum() < .4 * max(a['inside'].sum(), b['inside'].sum()):
            continue
        delta = cv2.absdiff(a['print_view'], b['print_view'][rotation, rotation])
        scores.append(float((delta[overlap].max(axis=1) > 32).mean()))
    return min(scores, default=None)


def plan(socks, boxes, left, inspect):
    ids = [sock['id'] for sock in socks]
    matches = set()
    uncertain = {sock['id'] for sock in socks if not sock['clear']}
    for a, b in itertools.combinations(socks, 2):
        pair = (a['id'], b['id'])
        status = classify(a, b)
        if status == 'different':
            continue
        verdict = inspect(pair, status)
        if verdict == 'm' and status == 'inspect_print':
            matches.add(pair)
        elif verdict != 'd':
            uncertain.update(pair)
    groups = []
    pending = set(ids)
    while pending:
        first = min(pending)
        group = [first]
        pending.remove(first)
        for candidate in sorted(pending):
            if all((member, candidate) in matches for member in group):
                group.append(candidate)
                pending.remove(candidate)
        groups.append(group)

    # Hold conflicting and unresolved groups before assigning any destination.
    membership = {sock: group[0] for group in groups for sock in group}
    for a, b in matches:
        if membership[a] != membership[b]:
            uncertain.update((a, b))
    placements = []
    held = []
    next_box = 0
    for group in groups:
        if uncertain.intersection(group):
            held.append({'socks': group, 'reason': 'reinspect'})
        elif len(group) == 1:
            placements.append({'socks': group, 'destination': 'left', 'region': left})
        elif next_box == len(boxes):
            held.append({'socks': group, 'reason': 'no empty right box'})
        else:
            placements.append({'socks': group, 'destination': f'box_{next_box + 1}',
                               'region': boxes[next_box]})
            next_box += 1
    return placements, held


def validate_layout(layout, shape):
    pickup, left, boxes = layout['pickup'], layout['left'], layout['boxes']
    if len(boxes) != 4:
        raise ValueError('Mark exactly four right-hand boxes in the configured order.')
    height, width = shape[:2]
    regions = [pickup, left, *boxes]
    for x, y, w, h in regions:
        if min(x, y) < 0 or min(w, h) <= 0 or x+w > width or y+h > height:
            raise ValueError('All regions must fit inside the overhead frame.')
    for (x, y, w, h), (a, b, aw, bh) in itertools.combinations(regions, 2):
        if max(x, a) < min(x+w, a+aw) and max(y, b) < min(y+h, b+bh):
            raise ValueError('Taped regions must not overlap.')
    if left[0]+left[2] > pickup[0] or any(box[0] < pickup[0]+pickup[2] for box in boxes):
        raise ValueError('The singles region must be left of pickup and all boxes to its right.')
    return pickup, left, boxes


def read_frame(args, parser):
    if args.image:
        return cv2.imread(str(args.image))
    camera = cv2.VideoCapture(args.camera)
    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        for _ in range(15):
            ok, frame = camera.read()
            if not ok:
                parser.error('The overhead camera stopped sending frames.')
        return frame
    finally:
        camera.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', type=Path)
    source.add_argument('--camera', type=int)
    parser.add_argument('--empty', required=True, type=Path)
    parser.add_argument('--layout', required=True, type=Path)
    parser.add_argument('--annotated', required=True, type=Path)
    parser.add_argument('--review', action='store_true', help='Compare pairs interactively')
    parser.add_argument('--flip-horizontal', action='store_true',
                        help='Use mirrored planning coordinates for a reversed overhead view')
    args = parser.parse_args()
    frame = read_frame(args, parser)
    empty = cv2.imread(str(args.empty))
    if frame is None or empty is None or frame.shape != empty.shape:
        parser.error('The reference and overhead frame must be readable and the same size.')
    if args.flip_horizontal:
        frame = cv2.flip(frame, 1)
        empty = cv2.flip(empty, 1)
    try:
        layout = json.loads(args.layout.read_text())
        pickup, left, boxes = validate_layout(layout, frame.shape)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    socks = detect(frame, empty, pickup)
    annotated = frame.copy()
    for index, (x, y, w, h) in enumerate([left, *boxes]):
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (255, 190, 0), 2)
        label = 'Singles' if index == 0 else f'Box {index}'
        cv2.putText(annotated, label, (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 190, 0), 2)
    by_id = {sock['id']: sock for sock in socks}
    for sock in socks:
        x, y, w, h = sock['bbox']
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(annotated, str(sock['id']), (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
    if not cv2.imwrite(str(args.annotated), annotated):
        parser.error('Could not save the annotated frame.')

    def inspect(pair, status):
        if not args.review:
            return 'u'
        crops = []
        for sock_id in pair:
            x, y, w, h = by_id[sock_id]['bbox']
            crop = frame[y:y+h, x:x+w]
            if args.flip_horizontal:
                crop = cv2.flip(crop, 1)
            scale = min(350 / w, 350 / h)
            crop = cv2.resize(crop, (max(1, round(w*scale)), max(1, round(h*scale))))
            crops.append(cv2.copyMakeBorder(crop, 0, 350-crop.shape[0], 0, 350-crop.shape[1],
                                            cv2.BORDER_CONSTANT))
        title = f'Socks {pair[0]} and {pair[1]} | D different, U recheck'
        if status == 'inspect_print':
            title += ', M identical'
            discrepancy = print_difference(*(by_id[sock_id] for sock_id in pair))
            if discrepancy is not None:
                title += f' | visible difference {discrepancy:.0%}'
        cv2.imshow(title, np.hstack(crops))
        try:
            while True:
                key = cv2.waitKey(100) & 255
                if key in (ord('u'), ord('d'), 27) or (key == ord('m') and status == 'inspect_print'):
                    return chr(key)
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    return 'u'
        finally:
            cv2.destroyAllWindows()

    placements, held = plan(socks, boxes, left, inspect)
    print(json.dumps({
        'frame_size': [frame.shape[1], frame.shape[0]],
        'flip_horizontal': args.flip_horizontal,
        'detections': [{'sock': s['id'], 'bbox': s['bbox']} for s in socks],
        'placements': placements, 'held': held,
    }, indent=2))


if __name__ == '__main__':
    main()
