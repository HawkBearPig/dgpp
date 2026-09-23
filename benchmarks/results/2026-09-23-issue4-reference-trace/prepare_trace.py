#!/usr/bin/env python3
"""Prepare a read-only overlay against the retained immutable image source."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

record = Path(__file__).resolve().parent
source = record.parent / '2026-09-23-issue4-reference/raw/image-source/models/qwen3_8_flash_next/nvidia/model.py'
original = source.read_text()
modified = original


def replace(old, new):
    global modified
    assert modified.count(old) == 1, old
    modified = modified.replace(old, new)


replace('from torch import nn\n', 'from torch import nn\nfrom . import issue4_trace as _issue4\n')
replace('        if self.layer_type == "linear_attention":\n            attn_out =',
        '        _issue4.capture("attn_in", block_input)\n'
        '        _issue4.capture("attn_injection", injection)\n'
        '        if self.layer_type == "linear_attention":\n            attn_out =')
replace('        mlp_hc = self.mlp_hyper_connection\n',
        '        _issue4.capture("attn_out", attn_out)\n'
        '        mlp_hc = self.mlp_hyper_connection\n')
replace('        mlp_out = self.mlp(block_input)\n        return hidden_states, mlp_out, injection\n',
        '        _issue4.capture("mlp_in", block_input)\n'
        '        _issue4.capture("residual", hidden_states)\n'
        '        _issue4.capture("mlp_injection", injection)\n'
        '        mlp_out = self.mlp(block_input)\n'
        '        _issue4.capture("mlp_out", mlp_out)\n'
        '        return hidden_states, mlp_out, injection\n')
replace('    ) -> torch.Tensor | IntermediateTensors:\n        if get_pp_group().is_first_rank:',
        '    ) -> torch.Tensor | IntermediateTensors:\n'
        '        _issue4.begin(input_ids, positions)\n'
        '        _issue4.capture("ngram_context", ngram_context)\n'
        '        _issue4.capture("query_start_loc", query_start_loc)\n'
        '        if get_pp_group().is_first_rank:')
replace('        block_output = None\n        injection = None\n        last_layer = None\n',
        '        _issue4.capture("embedding", hidden_states)\n'
        '        block_output = None\n        injection = None\n        last_layer = None\n')
replace('            last_layer = layer\n            hidden_states, block_output, injection = layer(',
        '            last_layer = layer\n            _issue4.set_layer(layer_idx)\n'
        '            hidden_states, block_output, injection = layer(')
replace('        return sample_hidden_states\n',
        '        _issue4.capture("final_hidden", sample_hidden_states)\n'
        '        return sample_hidden_states\n')
ast.parse(modified)
ast.parse((record / 'issue4_trace.py').read_text())
(record / 'raw/model.py').write_text(modified)
(record / 'diagnostic.patch').write_text(''.join(difflib.unified_diff(
    original.splitlines(keepends=True), modified.splitlines(keepends=True),
    fromfile='a/vllm/models/qwen3_8_flash_next/nvidia/model.py',
    tofile='b/vllm/models/qwen3_8_flash_next/nvidia/model.py')))
(record / 'source-provenance.json').write_text(json.dumps({
    'image': 'sha256:d464f3b466fa9c45ddbff8a812e80564503b6879a9fd95c1a47514f3f0df5a4a',
    'base_model_sha256': hashlib.sha256(original.encode()).hexdigest(),
    'overlay_model_sha256': hashlib.sha256(modified.encode()).hexdigest(),
    'trace_module_sha256': hashlib.sha256((record / 'issue4_trace.py').read_bytes()).hexdigest(),
}, indent=2) + '\n')
