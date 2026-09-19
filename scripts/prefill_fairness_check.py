#!/usr/bin/env python3
"""Verify a long cold prefill yields to an active streamed decode on an idle server."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
import urllib.request
import uuid

from vision_api_check import png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:18080')
    parser.add_argument('--images', action='store_true', help='include images in the cold prefill')
    parser.add_argument('--image-count', type=int, default=2, help='full-size images when --images is set')
    parser.add_argument('--decoders', type=int, choices=(1, 2), default=1,
                        help='two active streams exercise padded batched decode graphs')
    args = parser.parse_args()
    if not 1 <= args.image_count <= 64:
        parser.error('--image-count must be in [1, 64]')

    def get(path):
        with urllib.request.urlopen(args.url + path, timeout=10) as response:
            return json.load(response)

    initial = get('/metrics')
    assert initial['scheduler']['active'] == initial['scheduler']['queued'] == 0, initial
    assert initial['service']['admission']['prefill_budget_tokens'] > 0, initial
    model = get('/v1/models')['data'][0]['id']
    ready = threading.Event()
    dummy_started = threading.Event()
    event_times = [[] for _ in range(args.decoders)]

    def run(messages, maximum, record=None):
        body = {'model': model, 'messages': messages, 'max_tokens': maximum,
                'temperature': 0, 'reasoning_effort': 'low', 'prefix_cache': False,
                'stream': True, 'stream_options': {'include_usage': True, 'include_obfuscation': False}}
        request = urllib.request.Request(args.url + '/v1/chat/completions',
            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        usage = None
        done = False
        text = []
        with urllib.request.urlopen(request, timeout=300) as response:
            for raw in response:
                line = raw.decode().strip()
                if line == 'data: [DONE]':
                    done = True
                if not line.startswith('data: {'):
                    continue
                event = json.loads(line[6:])
                assert 'error' not in event, event
                if event.get('usage'):
                    usage = event['usage']
                text.extend(c.get('delta', {}).get('content', '') for c in event.get('choices', []))
                if record is not None and any(c.get('delta', {}).get('content') or
                                  c.get('delta', {}).get('reasoning_content') for c in event.get('choices', [])):
                    if record == -1:
                        dummy_started.set()
                    else:
                        event_times[record].append(time.monotonic())
                        if all(len(times) >= 3 for times in event_times):
                            ready.set()
        assert done and usage is not None, (done, usage)
        if record is None:
            assert 'ready' in ''.join(text).lower(), ''.join(text)
        return usage

    with ThreadPoolExecutor(max_workers=args.decoders + 1) as pool:
        # Leave a hole below both active slots. The subsequent prefill owns
        # slot 0 while the batched graph pads it and decodes slots 1 and 2.
        dummy = None
        if args.decoders == 2:
            dummy = pool.submit(run, [{'role': 'user', 'content':
                'Count from 1 to 1000, one number per line. Do not stop early.'}], 64, -1)
            deadline = time.monotonic() + 60
            while not dummy_started.wait(0.1):
                if dummy.done():
                    dummy.result()
                    raise AssertionError('slot-0 setup did not stream')
                assert time.monotonic() < deadline, 'slot-0 setup timed out'
        decodes = [pool.submit(run, [{'role': 'user', 'content':
            'Count from 1 to 1000, one number per line. Continue until you reach 1000. '
            'Do not summarize or skip numbers.'}], 1024, i) for i in range(args.decoders)]
        deadline = time.monotonic() + 60
        while not ready.wait(0.1):
            for i, decode in enumerate(decodes):
                if decode.done():
                    decode.result()
                    if len(event_times[i]) < 3:
                        raise AssertionError('decode ended before enough streaming events')
            assert time.monotonic() < deadline, 'decode did not begin streaming'
        if dummy is not None:
            dummy.result(timeout=60)
        marker = uuid.uuid4().hex
        content = [{'type': 'text', 'text': marker + '\n' +
            'Context record. ' * 7000 + '\nReply with the word ready.'}]
        if args.images:
            for i in range(args.image_count):
                color = (255, 0, 0) if i % 2 == 0 else (0, 0, 255)
                content.insert(0, {'type': 'image_url', 'image_url': {'url':
                    'data:image/png;base64,' + base64.b64encode(png(color, 896, 896)).decode()}})
        submitted_at = time.monotonic()
        long = pool.submit(run, [{'role': 'user', 'content': content}], 32)
        start = finish = None
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            metrics = get('/metrics')
            assert not metrics['service']['engine_failed'], metrics
            if metrics['scheduler']['prefilling']:
                if args.decoders == 2 and metrics['prefill']['requests']:
                    assert metrics['prefill']['requests'][0]['slot'] == 0, metrics['prefill']
                if start is None:
                    start = time.monotonic()
                finish = time.monotonic()
            elif start is not None:
                break
            if long.done():
                long.result()
                break
            time.sleep(0.05)
        assert start is not None and finish > start, 'no resumable prefill observed'
        during = [[t for t in times if start <= t <= finish] for times in event_times]
        assert all(len(times) >= 3 for times in during), ('active decode stalled during prefill',
                                                        [len(times) for times in during], finish - start)
        prefill_usage = long.result(timeout=180)
        decode_usage = [decode.result(timeout=180) for decode in decodes]
        assert prefill_usage['prompt_tokens_details']['cached_tokens'] == 0, prefill_usage
        gaps = [b - a for times in during for a, b in zip(times, times[1:])]
        admission_gaps = [b - a for times in event_times for a, b in zip(times, times[1:])
                          if b >= submitted_at and a <= finish]
        print(json.dumps({'images': args.images, 'decoders': args.decoders,
                          'image_count': args.image_count if args.images else 0,
                          'decode_events_during_prefill': sum(map(len, during)),
                          'per_stream_events_during_prefill': list(map(len, during)),
                          'prefill_observed_s': finish - start,
                          'max_decode_gap_during_prefill_s': max(gaps),
                          'max_decode_gap_including_admission_s': max(admission_gaps),
                          'prefill_usage': prefill_usage, 'decode_usage': decode_usage}), flush=True)
    final = get('/metrics')
    assert not final['service']['engine_failed'], final
    assert final['scheduler']['prefilling'] == 0, final
    print('prefill fairness check passed')


if __name__ == '__main__':
    main()
