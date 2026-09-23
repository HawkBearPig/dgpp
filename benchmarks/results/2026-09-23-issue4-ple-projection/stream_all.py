#!/usr/bin/env python3
"""Stream every native PLE row through the independent checkpoint-to-gate oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from replay import bf, rb, norm, gate, sha


class Error:
    def __init__(self):
        self.n = self.exact = self.mismatch = self.nonfinite = 0
        self.reference = self.actual = self.error = self.max_abs = 0.0
        self.worst = []

    def add(self, expected, actual, begin):
        count = expected.shape[0]
        expected = expected.astype(np.float64).reshape(count, -1)
        actual = actual.astype(np.float64).reshape(expected.shape)
        difference = np.abs(expected-actual)
        ref2 = np.sum(expected**2, axis=1)
        got2 = np.sum(actual**2, axis=1)
        err2 = np.sum(difference**2, axis=1)
        relative = np.sqrt(err2 / np.maximum(np.maximum(ref2,got2),1e-60))
        mismatched = difference > np.maximum(1e-7,np.maximum(np.abs(expected),np.abs(actual))*(2/128))
        self.n += expected.size
        self.exact += int((difference==0).sum())
        self.mismatch += int(mismatched.sum())
        self.nonfinite += int((~np.isfinite(expected)|~np.isfinite(actual)).sum())
        self.reference += float(ref2.sum())
        self.actual += float(got2.sum())
        self.error += float(err2.sum())
        self.max_abs = max(self.max_abs, float(difference.max()))
        for row in np.argsort(relative)[-10:]:
            self.worst.append(dict(position=begin+int(row), relative_l2=float(relative[row]),
                                   max_abs=float(difference[row].max()),mismatches=int(mismatched[row].sum())))
        self.worst = sorted(self.worst,key=lambda x:x['relative_l2'],reverse=True)[:20]

    def result(self):
        return dict(elements=self.n,relative_l2=float(np.sqrt(self.error/max(self.reference,self.actual,1e-60))),
                    max_abs=self.max_abs,bitwise_equal_elements=self.exact,two_bf16_relative_mismatches=self.mismatch,
                    mismatch_fraction=self.mismatch/max(self.n,1),nonfinite=self.nonfinite,worst_rows=self.worst)


def main(args):
    begin_time=time.monotonic()
    provenance=json.loads(args.weights.with_suffix('.json').read_text())
    assert sha(args.weights)==provenance['bundle_sha256']
    manifest={x['path']:x for x in json.loads(args.manifest.read_text())['entries']}
    weight=dict(np.load(args.weights,allow_pickle=False))
    for key in ('key_proj0','key_proj1','value_proj0','value_proj1'):
        weight[key]=weight[key].astype(np.float64).T.copy()
    streams,states={},{}
    tokens=261290
    for rank in (0,1):
        for field,width in (('ple_embedding',1280),('ple_residual_before',10240),('ple_gv',10240),('ple_un',10240)):
            relative=f'rank{rank}/layer1/{field}.bin'
            path=args.archive/relative
            stat=path.stat()
            assert stat.st_size==manifest[relative]['bytes']==tokens*width*2
            states[relative]=(stat.st_ino,stat.st_size,stat.st_mtime_ns)
            streams[(rank,field)]=(path.open('rb'),width)
    checks={name:Error() for name in ('checkpoint_to_gate','checkpoint_to_conv_norm','isolated_conv_norm')}
    hashes={str(key):hashlib.sha256() for key in streams}
    minimum,maximum=1.0,0.0
    for start in range(0,tokens,args.batch):
        end=min(start+args.batch,tokens)
        current={}
        for key,(stream,width) in streams.items():
            payload=stream.read((end-start)*width*2)
            assert len(payload)==(end-start)*width*2
            hashes[str(key)].update(payload)
            bits=np.frombuffer(payload,dtype='<u2').reshape(end-start,width)
            current[key]=bf(bits)
        for field in ('ple_residual_before','ple_gv','ple_un'):
            assert np.array_equal(current[(0,field)],current[(1,field)]), (start,field)
        projections={}
        for name in ('key_proj','value_proj'):
            parts=[rb(current[(rank,'ple_embedding')].astype(np.float64) @ weight[name+str(rank)]) for rank in (0,1)]
            projections[name]=rb(parts[0]+parts[1])
        key=norm(projections['key_proj'],weight['norm_key'],1e-6)
        query=norm(current[(0,'ple_residual_before')],weight['norm_query'],1e-6)
        gated,gates=gate(key,query,projections['value_proj'])
        minimum=min(minimum,float(gates.min())); maximum=max(maximum,float(gates.max()))
        checks['checkpoint_to_gate'].add(gated,current[(0,'ple_gv')],start)
        normalized=norm(gated,weight['norm_conv'],1e-6)
        checks['checkpoint_to_conv_norm'].add(normalized,current[(0,'ple_un')],start)
        checks['isolated_conv_norm'].add(norm(current[(0,'ple_gv')],weight['norm_conv'],1e-6),current[(0,'ple_un')],start)
        if start==0 or end==tokens or start % (args.batch*16)==0:
            progress=dict(rows_completed=end,total_rows=tokens,seconds=time.monotonic()-begin_time,
                          checks={name:check.result() for name,check in checks.items()},
                          peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            args.output.with_suffix('.progress.json').write_text(json.dumps(progress,indent=2)+'\n')
            print(end,'/',tokens,'rows',round(progress['seconds'],2),'seconds',flush=True)
    source_hashes=[]
    for (rank,field),(stream,width) in streams.items():
        assert stream.read(1)==b''
        stream.close()
        relative=f'rank{rank}/layer1/{field}.bin'
        stat=(args.archive/relative).stat()
        assert states[relative]==(stat.st_ino,stat.st_size,stat.st_mtime_ns)
        checksum=hashes[str((rank,field))].hexdigest()
        assert checksum==manifest[relative]['sha256'],relative
        source_hashes.append(dict(file=relative,bytes=stat.st_size,sha256=checksum,exact=True))
    report=dict(completed=True,tokens=tokens,checks={name:check.result() for name,check in checks.items()},
                seconds=time.monotonic()-begin_time,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                source_hashes=source_hashes,source_files_fully_rehashed=True,weights=provenance,
                gate_range=[minimum,maximum],rank_replicated_fields_equal=True,
                script_sha256=sha(Path(__file__)),oracle_sha256=sha(Path(__file__).with_name('replay.py')),
                scope='All 261290 rows from the original native NVFP4 forced-prefix capture; captured embeddings and pre-PLE residuals are the inputs. Does not establish end-to-end model correctness.')
    assert all(x['nonfinite']==0 for x in report['checks'].values())
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('Complete; source content hashes verified',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch',type=int,default=512)
    main(parser.parse_args())
