#!/usr/bin/env python3
"""Every __global__ kernel under src/ carries dgpp_pdl_wait() as its first statement (kernels/pdl.cuh), or opens with
dgpp_pdl_preamble() and carries the wait further down its body (a kernel that loads its own weights first): the decode
graph's post-pass (engine/graph_pdl.hpp) turns every kernel-to-kernel edge into a programmatic dependency for kernels in
our namespace, and a kernel without the wait would read its predecessor's output before it is written (2026-10-20).
Usage: pdl_wait_audit.py [repository root]; exits 1 naming each kernel without the wait.
What a preamble may touch is the header's rule and a reviewer's check: this audit holds only that the wait is there."""
import glob, os, re, sys

IDENT = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
def kernel_defs(s):
    """Yields (name, body_brace_index) for every __global__ kernel DEFINITION in source text s."""
    out = []
    for m in re.finditer(r'__global__(?=\s+(?:(?:inline|static)\s+)*(?:void\b|__launch_bounds__\b))', s):  # a definition's head, not a mention in prose
        i = m.end()
        name = None
        while True:
            p = s.find('(', i)
            if p < 0: break
            # the identifier right before '('
            q = p
            while q > 0 and s[q-1].isspace(): q -= 1
            r = q
            while r > 0 and (s[r-1].isalnum() or s[r-1] == '_'): r -= 1
            ident = s[r:q]
            # a ';' or '{' before this '(' ends the search (a declaration or something else)
            seg = s[i:r]
            if ';' in seg or '{' in seg: break
            # skip balanced parens
            depth, j = 1, p + 1
            while depth and j < len(s):
                depth += (s[j] == '(') - (s[j] == ')'); j += 1
            if ident == '__launch_bounds__' or ident == '' or ident in ('__attribute__',):
                i = j; continue
            name = ident
            # after the parameter list: '{' before any ';' = a definition
            rest = s[j:j + 400]
            b = rest.find('{'); semi = rest.find(';')
            if b >= 0 and (semi < 0 or b < semi): out.append((name, j + b))
            break
    return out

def body_of(s, b):
    """The text of the braces opening at s[b]."""
    depth, j = 1, b + 1
    while depth and j < len(s):
        depth += (s[j] == '{') - (s[j] == '}'); j += 1
    return s[b + 1:j - 1]

root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
missing, total, preamble = [], 0, 0
for f in sorted(glob.glob(os.path.join(root, 'src', '**', '*.cu'), recursive=True) + glob.glob(os.path.join(root, 'src', '**', '*.cuh'), recursive=True)):
    s = open(f).read()
    for name, b in kernel_defs(s):
        total += 1
        head = s[b + 1:b + 80].lstrip()
        if head.startswith('dgpp_pdl_wait();'): continue
        # A kernel that loads its own weights before the wait opens with the marker; the wait follows in its body.
        if head.startswith('dgpp_pdl_preamble();') and 'dgpp_pdl_wait();' in body_of(s, b):
            preamble += 1
            continue
        missing.append(f"{os.path.relpath(f, root)}: {name}")
if missing:
    print("kernels without dgpp_pdl_wait() as their first statement:")
    for x in missing: print("  " + x)
    sys.exit(1)
print(f"pdl_wait_audit: {total} kernels carry the wait ({preamble} behind a preamble)")
