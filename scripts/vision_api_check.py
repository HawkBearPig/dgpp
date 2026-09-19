#!/usr/bin/env python3
"""Exercise GLM image input through the live OpenAI-compatible API.

Uses generated PNG fixtures and the Python standard library. The model must
identify colors from pixels, with identical text across differently colored
images; repeated and concurrent requests check isolation from the prefix cache.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import struct
import urllib.request
import zlib


def png(color, width=112, height=112):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    pixels = b''.join(b'\0' + bytes(color) * width for _ in range(height))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b''))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:18080')
    parser.add_argument('--model', help='defaults to the first served model')
    args = parser.parse_args()
    with urllib.request.urlopen(args.url + '/v1/models', timeout=10) as response:
        models = json.load(response)['data']
    if not args.model:
        args.model = models[0]['id']
    assert any(m['id'] == args.model and 'image' in m.get('input_modalities', []) for m in models)

    def run(colors, names, stream=False, n=1, width=112, height=112, prefix=''):
        content = [{'type': 'text', 'text': prefix + 'What is the dominant color of each image? '
                    'Answer only the color names, in image order.'}]
        content += [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' +
                    base64.b64encode(png(color, width, height)).decode()}} for color in colors]
        body = {'model': args.model, 'messages': [{'role': 'user', 'content': content}],
                'temperature': 0, 'max_tokens': 256, 'reasoning_effort': 'low', 'stream': stream, 'n': n}
        if stream:
            body['stream_options'] = {'include_usage': True, 'include_obfuscation': False}
        request = urllib.request.Request(args.url + '/v1/chat/completions', data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read().decode()
        if stream:
            assert 'data: [DONE]' in raw
            events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: {')]
            texts = [''.join(c['delta'].get('content', '') for e in events for c in e['choices'])]
            usage = [e['usage'] for e in events if e.get('usage')][-1]
        else:
            reply = json.loads(raw)
            texts = [c['message']['content'] for c in reply['choices']]
            assert len(texts) == n
            usage = reply['usage']
        assert 0 <= usage['prompt_tokens_details']['cached_tokens'] < usage['prompt_tokens']
        for text in texts:
            cursor = 0
            for name in names:
                cursor = text.lower().find(name, cursor)
                assert cursor >= 0, (names, text)
                cursor += len(name)
        print(json.dumps({'expected': names, 'answers': texts, 'stream': stream, 'usage': usage}), flush=True)

    red, blue, green = (255, 0, 0), (0, 0, 255), (0, 255, 0)
    run([red], ['red'])
    run([blue], ['blue'])
    run([red, blue], ['red', 'blue'])
    run([green], ['green'], stream=True)
    run([blue], ['blue'], n=2)
    run([red], ['red'], width=1000, height=900)
    # Put the image beyond a text-prefill chunk boundary, including MTP's
    # shifted embedding input and the subsequent chunk's state.
    run([green], ['green'], prefix='Context. ' * 1200)
    # A 1024-token image beginning around row 1500 straddles the 2048-row
    # GLM prefill boundary, rather than merely starting in a later chunk.
    run([red], ['red'], width=1000, height=900, prefix='Context. ' * 750)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(run, [red], ['red']), pool.submit(run, [blue], ['blue'])]
        for job in jobs:
            job.result()
    print('vision API checks passed')


if __name__ == '__main__':
    main()
