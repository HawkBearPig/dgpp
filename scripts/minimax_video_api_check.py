#!/usr/bin/env python3
"""MiniMax video task, cache and plain/speculative API checks on an idle world.

Requires FFmpeg with libx264rgb/libvpx-vp9 to generate lossless clips. Server
decoding itself uses native libraries. Output is a diagnostic artifact, not a
throughput result. Keep it outside the tracked tree.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import urllib.error
import urllib.request
import uuid

from vision_api_check import png


def clip(colors, width=112, height=112, container='mp4'):
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-threads', '2',
               '-filter_threads', '1', '-f', 'rawvideo', '-pixel_format', 'rgb24',
               '-video_size', f'{width}x{height}', '-framerate', '1', '-i', 'pipe:0', '-an']
    command += (['-c:v', 'libx264rgb', '-crf', '0', '-preset', 'veryfast',
                 '-pix_fmt', 'rgb24', '-movflags', 'frag_keyframe+empty_moov'] if container == 'mp4'
                else ['-c:v', 'libvpx-vp9', '-lossless', '1', '-pix_fmt', 'gbrp'])
    command += ['-threads', '2', '-fflags', '+bitexact', '-flags:v', '+bitexact', '-f', container, 'pipe:1']
    pixels = b''.join(bytes(color) * width * height for color in colors)
    result = subprocess.run(command, input=pixels, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=True, timeout=60)
    return {'type': 'video_url', 'video_url': {
        'url': f'data:video/{container};base64,' + base64.b64encode(result.stdout).decode(), 'fps': 1}}


def text(value):
    return {'type': 'text', 'text': value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:18080')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--skip-cache', action='store_true')
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error('--concurrency must be positive')
    with urllib.request.urlopen(args.url + '/v1/models', timeout=10) as response:
        model = json.load(response)['data'][0]
    assert 'video' in model.get('input_modalities', []), model
    red, blue, green = (255, 0, 0), (0, 0, 255), (0, 255, 0)
    solid_red, solid_blue = clip([red] * 4), clip([blue] * 4)
    red_blue, blue_red = clip([red] * 4 + [blue] * 4), clip([blue] * 6 + [red] * 4)
    large_blue = clip([red] * 4 + [blue] * 4, 896, 672)
    large_green = clip([red] * 4 + [green] * 4, 896, 672)
    still_green = {'type': 'image_url', 'image_url': {
        'url': 'data:image/png;base64,' + base64.b64encode(png(green)).decode()}}
    color = 'What is the dominant color throughout the video? Answer only its color name.'
    order = 'Name the dominant color at the beginning and at the end of the video, in that order. Answer only the two color names.'
    last = 'What is the dominant color in the last frame of the video? Answer only its color name.'
    cases = [
        ('red', [solid_red, text(color)], ['red']),
        ('blue', [solid_blue, text(color)], ['blue']),
        ('red-then-blue', [red_blue, text(order)], ['red', 'blue']),
        ('cross-segment-order', [blue_red, text(order)], ['blue', 'red']),
        ('large-chunk-crossing', [text('Context. ' * 260), large_blue, text(order)], ['red', 'blue']),
        ('mixed-image-video', [still_green, red_blue, text('Name the color of the still image, then the last video frame. Answer only the two color names.')], ['green', 'blue']),
        ('webm', [clip([blue] * 4, container='webm'), text(color)], ['blue']),
        ('odd-final-frame', [clip([red] * 2 + [blue] * 2 + [green]), text(last)], ['green']),
    ]

    def request(messages, stream=False, cache=False):
        body = {'model': model['id'], 'messages': messages, 'temperature': 0,
                'reasoning_effort': 'low', 'max_tokens': 256, 'prefix_cache': cache, 'stream': stream}
        if stream:
            body['stream_options'] = {'include_usage': True, 'include_obfuscation': False}
        req = urllib.request.Request(args.url + '/v1/chat/completions', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=600) as response:
            raw = response.read().decode()
        if stream:
            assert 'data: [DONE]' in raw
            events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: {')]
            content = ''.join(c['delta'].get('content', '') for event in events for c in event['choices'])
            result = {'choices': [{'message': {'content': content}}],
                      'usage': [event['usage'] for event in events if event.get('usage')][-1]}
        else:
            result = json.loads(raw)
        return {'request': body, 'response': result}

    def answer(result, expected):
        value = result['response']['choices'][0]['message']['content'].lower()
        position = 0
        for word in expected:
            found = value.find(word, position)
            assert found >= 0, (expected, result['response'])
            position = found + len(word)
        return value

    def cached(result):
        return result['response']['usage']['prompt_tokens_details']['cached_tokens']

    def run(case):
        name, parts, expected = case
        result = request([{'role': 'user', 'content': parts}])
        print(name, repr(answer(result, expected)), flush=True)
        assert cached(result) == 0, result['response']['usage']
        return name, result

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = dict(pool.map(run, cases))
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps({'cases': results, 'cache': {}, 'complete': False}, indent=2) + '\n')
    if args.compare:
        reference = json.loads(args.compare.read_text())['cases']
        assert results.keys() == reference.keys()
        for name, result in results.items():
            assert result['request'] == reference[name]['request'], name
            x, y = result['response'], reference[name]['response']
            assert x['choices'] == y['choices'], (name, x['choices'], y['choices'])
            for key in ('prompt_tokens', 'completion_tokens'):
                assert x['usage'][key] == y['usage'][key], (name, key, x['usage'], y['usage'])
        print('Plain/speculative choices and token counts match exactly.', flush=True)
    streamed = request([{'role': 'user', 'content': [solid_red, text(color)]}], stream=True)
    assert answer(streamed, ['red']) == answer(results['red'], ['red'])
    assert cached(streamed) == 0

    cache_results = {}
    if not args.skip_cache:
        messages = [{'role': 'system', 'content': 'Video cache regression ' + uuid.uuid4().hex},
                    {'role': 'user', 'content': [large_blue, text('Context. ' * 3000), text(last)]}]
        for name, cache in [('cold', True), ('warm', True), ('disabled', False)]:
            result = request(messages, cache=cache)
            answer(result, ['blue'])
            cache_results[name] = result
        cold, warm, disabled = [cache_results[key] for key in ('cold', 'warm', 'disabled')]
        assert cached(warm) > .95 * warm['response']['usage']['prompt_tokens']
        assert cached(disabled) == 0 and answer(warm, ['blue']) == answer(disabled, ['blue'])
        changed = json.loads(json.dumps(messages))
        changed[1]['content'][0] = large_green
        future = request(changed, cache=True)
        answer(future, ['green'])
        assert cached(future) < .1 * future['response']['usage']['prompt_tokens'], future['response']['usage']
        cache_results['future-frame-change'] = future
        continuation = messages + [{'role': 'assistant', 'content': answer(warm, ['blue'])},
                                   {'role': 'user', 'content': 'What color was the last video frame? Answer only its color name.'}]
        continued = request(continuation, stream=True, cache=True)
        answer(continued, ['blue'])
        assert cached(continued) >= cached(warm)
        cache_results['continued'] = continued
        print('Video cache reuse, future-frame invalidation and streaming continuation passed.', flush=True)

    bad = {'model': model['id'], 'messages': [{'role': 'user', 'content': [
        {'type': 'video_url', 'video_url': {'url': 'data:video/mp4;base64,AAAA'}}]}]}
    req = urllib.request.Request(args.url + '/v1/chat/completions', data=json.dumps(bad).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        urllib.request.urlopen(req, timeout=30)
        raise AssertionError('invalid video accepted')
    except urllib.error.HTTPError as error:
        detail = json.load(error)['error']
        assert error.code == 400 and detail['code'] == 'invalid_video', detail
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps({'cases': results, 'cache': cache_results, 'complete': True}, indent=2) + '\n')
    print('MiniMax video API checks passed.', flush=True)


if __name__ == '__main__':
    main()
