"""Audit a LeRobot v3 sock dataset and optionally train ACT using audited episodes."""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq


TASK = 'Pick up a sock and place it on the left side'


def audit(root, previews):
    info = json.loads((root / 'meta/info.json').read_text())
    if info['codebase_version'] != 'v3.0':
        raise ValueError('Expected a LeRobot v3.0 dataset.')
    fps = info['fps']
    cameras = [key for key, feature in info['features'].items() if feature['dtype'] == 'video']
    if not cameras:
        raise ValueError('Dataset has no video streams.')
    task_table = pq.read_table(root / 'meta/tasks.parquet').to_pandas()
    tasks = task_table['task_index'].to_dict()
    episodes = [ep for path in sorted((root / 'meta/episodes').rglob('*.parquet'))
                for ep in pq.read_table(path).to_pylist()]
    ids = [episode['episode_index'] for episode in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate episode indices in dataset metadata.')
    if previews:
        previews.mkdir(parents=True, exist_ok=False)
    data_files = {}
    video_files = {}
    results = []
    for episode in episodes:
        index = episode['episode_index']
        length = episode['length']
        errors = []
        try:
            data_path = root / info['data_path'].format(
                chunk_index=episode['data/chunk_index'], file_index=episode['data/file_index'])
            if data_path not in data_files:
                data_files[data_path] = pq.read_table(data_path).to_pandas()
            data = data_files[data_path]
            rows = data[data.episode_index == index]
            if (length <= 0 or len(rows) != length
                    or not np.array_equal(rows.frame_index, np.arange(length))
                    or not np.array_equal(rows['index'], np.arange(
                        episode['dataset_from_index'], episode['dataset_to_index']))
                    or not np.allclose(rows.timestamp, np.arange(length)/fps, atol=.002, rtol=0)):
                errors.append('Inconsistent frame indices, length or timestamps.')
            if episode['tasks'] != [TASK] or TASK not in tasks or not (rows.task_index == tasks[TASK]).all():
                errors.append('Unexpected task label or frame task index.')
            for key in ('observation.state', 'action'):
                if len(rows) and (np.stack(rows[key]).shape != (len(rows), 6)
                                  or not np.isfinite(np.stack(rows[key])).all()):
                    errors.append(f'Invalid joint values in {key}.')
        except (OSError, ValueError, KeyError) as error:
            errors.append(f'Frame data: {error}')
        for key in cameras:
            prefix = f'videos/{key}'
            try:
                path = root / info['video_path'].format(
                    video_key=key, chunk_index=episode[f'{prefix}/chunk_index'],
                    file_index=episode[f'{prefix}/file_index'])
                start, end = (episode[f'{prefix}/{field}_timestamp'] for field in ('from', 'to'))
                if not math.isclose(end-start, length/fps, abs_tol=1/fps):
                    raise ValueError('Episode video range disagrees with the frame count.')
                if path not in video_files:
                    with av.open(str(path)) as video:
                        stream = video.streams.video[0]
                        video_files[path] = (float(stream.duration * stream.time_base),
                                             stream.height, stream.width, float(stream.average_rate))
                duration, height, width, rate = video_files[path]
                if (start < 0 or end > duration + 1/fps or
                        [height, width, 3] != info['features'][key]['shape'] or abs(rate-fps) > .01):
                    raise ValueError('Video bounds, dimensions or FPS disagree with metadata.')
                with av.open(str(path)) as video:
                    stream = video.streams.video[0]
                    for label, target in [('start', start), ('middle', (start+end)/2),
                                          ('end', end-1/fps)]:
                        video.seek(round(target / stream.time_base), stream=stream)
                        frame = next((f for f in video.decode(stream)
                                      if f.time is not None and f.time >= target-.5/fps), None)
                        if frame is None or abs(frame.time-target) > 1/fps:
                            raise ValueError(f'Could not decode the {label} frame.')
                        if previews:
                            frame.to_image().save(previews / f'{index:02}_{key.rsplit(".", 1)[-1]}_{label}.jpg')
            except (OSError, ValueError, KeyError, IndexError, TypeError, av.FFmpegError) as error:
                errors.append(f'{key}: {error}')
        results.append({'episode': index, 'frames': length, 'errors': errors})
    return {'declared_episodes': info['total_episodes'], 'tasks': list(tasks),
            'cameras': cameras, 'episodes': results}


def evaluate(root, checkpoint, episodes, device, stride):
    """Compare each sampled first action with the recorded action and a no-motion baseline."""
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    saved_config = json.loads((checkpoint / 'train_config.json').read_text())
    trained_episodes = saved_config['dataset'].get('episodes')
    if not trained_episodes or set(trained_episodes).intersection(episodes):
        raise ValueError('The checkpoint does not establish that these episodes were held out.')
    policy = ACTPolicy.from_pretrained(checkpoint).to(device).eval()
    pre, post = make_pre_post_processors(
        policy.config, str(checkpoint),
        preprocessor_overrides={'device_processor': {'device': device}},
    )
    dataset = LeRobotDataset('local/socks-training-20', root=root, episodes=episodes,
                             video_backend='pyav')
    predictions, baseline = [], []
    with torch.inference_mode():
        for index in range(0, len(dataset), stride):
            row = dataset[index]
            inputs = {key: row[key] for key in policy.config.input_features}
            action = post(policy.predict_action_chunk(pre(inputs))[:, 0]).cpu()[0]
            predictions.append((action - row['action']).abs().numpy())
            baseline.append((row['observation.state'] - row['action']).abs().numpy())
    if not predictions:
        raise ValueError('No evaluation frames selected.')
    return {'holdout': episodes, 'samples': len(predictions),
            'action_mae_by_joint': np.mean(predictions, axis=0).tolist(),
            'no_motion_mae_by_joint': np.mean(baseline, axis=0).tolist()}


def train(command):
    from lerobot.scripts import lerobot_train

    if os.name == 'nt':
        update = lerobot_train.update_last_checkpoint

        def update_if_permitted(checkpoint):
            try:
                update(checkpoint)
            except OSError as error:
                if getattr(error, 'winerror', None) != 1314:
                    raise
                print(f'Windows blocked the last-checkpoint link. Use {checkpoint}.', file=sys.stderr)

        lerobot_train.update_last_checkpoint = update_if_permitted
    sys.argv[1:] = command
    lerobot_train.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--previews', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--train', type=Path, help='New checkpoint output directory')
    mode.add_argument('--evaluate', type=Path, help='Local pretrained_model checkpoint')
    parser.add_argument('--holdout', type=int, nargs='+')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--from-scratch', action='store_true')
    parser.add_argument('--stride', type=int, default=30)
    args = parser.parse_args()
    root = args.root.resolve()
    for output in (args.previews, args.train):
        if output and (output.exists() or output.resolve().is_relative_to(root)):
            parser.error('Output paths must be new and outside the source dataset.')
    if min(args.steps, args.batch_size, args.stride) < 1:
        parser.error('Steps, batch size and stride must be positive.')
    report = audit(root, args.previews)
    print(json.dumps(report, indent=2), flush=True)
    if args.train or args.evaluate:
        valid = {item['episode'] for item in report['episodes'] if not item['errors']}
        holdout = set(args.holdout or [])
        if not holdout or not holdout <= valid or (args.train and not valid - holdout):
            parser.error('Choose audited holdout episodes and leave at least one training episode.')
        import torch
        if args.device == 'cuda' and not torch.cuda.is_available():
            parser.error('CUDA unavailable. Use a CUDA host or explicitly select --device cpu.')
        if args.evaluate:
            if not (args.evaluate / 'model.safetensors').is_file():
                parser.error('Point --evaluate at a local pretrained_model directory.')
            print(json.dumps(evaluate(root, args.evaluate, sorted(holdout),
                                      args.device, args.stride), indent=2))
            return
        command = [
            '--dataset.repo_id=local/socks-training-20', f'--dataset.root={root}',
            f'--dataset.episodes={json.dumps(sorted(valid - holdout))}',
            '--dataset.video_backend=pyav', '--policy.type=act',
            f'--policy.device={args.device}', '--policy.push_to_hub=false',
            f'--output_dir={args.train.resolve()}', '--wandb.enable=false',
            f'--steps={args.steps}', f'--batch_size={args.batch_size}', '--num_workers=0',
        ]
        if args.from_scratch:
            command.append('--policy.pretrained_backbone_weights=null')
        train(command)


if __name__ == '__main__':
    main()
