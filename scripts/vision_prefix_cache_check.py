#!/usr/bin/env python3
"""Check image prefix reuse, continuation and pixel isolation on an idle GLM server."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import time
import urllib.request
import uuid

from vision_api_check import png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:18080')
    args = parser.parse_args()
    with urllib.request.urlopen(args.url + '/v1/models', timeout=10) as response:
        model = json.load(response)['data'][0]['id']
    marker = 'Cache regression ' + uuid.uuid4().hex

    def image(color, width=112, height=112):
        return {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' +
                base64.b64encode(png(color, width, height)).decode()}}

    def request(label, messages, expected, cache=True, stream=False):
        body = {'model': model, 'messages': messages, 'temperature': 0,
                'reasoning_effort': 'low', 'max_tokens': 256, 'prefix_cache': cache, 'stream': stream}
        if stream:
            body['stream_options'] = {'include_usage': True, 'include_obfuscation': False}
        started = time.monotonic()
        req = urllib.request.Request(args.url + '/v1/chat/completions',
                                     data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=300) as response:
            raw = response.read().decode()
        if stream:
            assert 'data: [DONE]' in raw, raw
            events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: {')]
            content = ''.join(c['delta'].get('content', '') for e in events for c in e['choices'])
            usage = [e['usage'] for e in events if e.get('usage')][-1]
        else:
            reply = json.loads(raw)
            content, usage = reply['choices'][0]['message']['content'], reply['usage']
        assert expected in content.lower(), (label, expected, content)
        result = {'case': label, 'answer': content, 'usage': usage, 'wall_s': time.monotonic() - started}
        print(json.dumps(result), flush=True)
        return result

    def cached(result):
        return result['usage']['prompt_tokens_details']['cached_tokens']

    red, blue, green = (255, 0, 0), (0, 0, 255), (0, 255, 0)
    question = {'type': 'text', 'text': 'Name the dominant color of the last image. Answer only its color name.'}
    messages = [{'role': 'system', 'content': marker},
                {'role': 'user', 'content': [image(red), {'type': 'text', 'text': 'Context. ' * 3000}, question]}]
    cold = request('long-image-cold', messages, 'red')
    warm = request('long-image-warm', messages, 'red')
    assert cached(warm) > 0.95 * warm['usage']['prompt_tokens'], warm
    control = request('long-image-cache-disabled', messages, 'red', cache=False)
    assert cached(control) == 0, control
    assert warm['answer'] == control['answer'], (warm, control)

    continuation = messages + [{'role': 'assistant', 'content': warm['answer']},
                               {'role': 'user', 'content': 'What color was that image? Answer only its color name.'}]
    turn = request('retired-image-continuation', continuation, 'red', stream=True)
    assert cached(turn) >= cached(warm), turn

    changed = json.loads(json.dumps(messages))
    changed[1]['content'][0] = image(blue)
    different = request('same-shape-different-pixels', changed, 'blue')
    assert cached(different) < 0.1 * different['usage']['prompt_tokens'], different

    # Attach before a new suffix image, then reuse a prefix containing both.
    appended = continuation + [{'role': 'assistant', 'content': turn['answer']},
                               {'role': 'user', 'content': [image(green), question]}]
    suffix = request('new-image-after-attach', appended, 'green')
    assert cached(suffix) >= cached(warm), suffix
    suffix_warm = request('two-image-repeat', appended, 'green')
    assert cached(suffix_warm) >= cached(suffix), suffix_warm

    # A large image spans a 2048-token chunk boundary; a changed prefix
    # forces the first pass through its image-aware cold path.
    crossing = [{'role': 'system', 'content': marker + ' crossing'},
                {'role': 'user', 'content': [{'type': 'text', 'text': 'Context. ' * 750},
                                            image(red, 1000, 900), question]}]
    request('image-across-chunk', crossing, 'red')
    cross_warm = request('image-across-chunk-repeat', crossing, 'red')
    assert cached(cross_warm) > 0.9 * cross_warm['usage']['prompt_tokens'], cross_warm

    # Both old history limits: more than eight images and 8192 visual tokens.
    history = [{'role': 'system', 'content': marker + ' extended history'}]
    for _ in range(12):
        history.append({'role': 'user', 'content': [image(blue, 896, 896)]})
        history.append({'role': 'assistant', 'content': 'Image received.'})
    history.append({'role': 'user', 'content': [question]})
    many = request('twelve-full-size-images', history, 'blue')
    assert many['usage']['prompt_tokens'] > 12288, many
    many_warm = request('twelve-image-history-reuse', history, 'blue')
    assert cached(many_warm) > 0.95 * many_warm['usage']['prompt_tokens'], many_warm
    request('twelve-image-history-cache-disabled', history, 'blue', cache=False)

    # Mixed small-image history checks ordering/content separately. The model
    # misidentifies the final color in 12 full-size mixed-image turns under
    # both streaming and an eager all-images reference; see the validation record.
    mixed = [{'role': 'system', 'content': marker + ' mixed history'}]
    for i in range(12):
        parts = [image(red if i < 11 else blue)]
        if i == 11:
            parts.append({'type': 'text', 'text': 'What color is the image in this message? '
                           'Answer only the color name.'})
        mixed.append({'role': 'user', 'content': parts})
        if i < 11:
            mixed.append({'role': 'assistant', 'content': 'Image received.'})
    request('twelve-mixed-small-images', mixed, 'blue', cache=False)

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(request, 'concurrent-red', messages, 'red'),
                pool.submit(request, 'concurrent-blue', changed, 'blue')]
        for job in jobs:
            result = job.result()
            assert cached(result) > 0.95 * result['usage']['prompt_tokens'], result
    print('vision prefix cache checks passed', flush=True)


if __name__ == '__main__':
    main()
