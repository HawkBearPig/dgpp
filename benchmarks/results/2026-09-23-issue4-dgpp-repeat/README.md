# Issue #4: DGPP within-world repeatability

The exact pinned vLLM reference changed from 5/6 to 6/6 on two identical
requests in one world, with the same cached tactics and prefix caching disabled.
This control repeats the unchanged original request twice on the retained clean
DGPP baseline, using the original model and engine configuration. Each request
must report zero cached tokens. It tests whether the same behavior occurs in
DGPP and could point to a serving-state/slot-reuse defect.

Both requests returned 5/6 with `val_5dac9ed720abddaf`, 184 completion tokens,
normal stop and zero cached tokens. Assistant text and usage are byte-identical.
Durations were 218.29 and 218.75 seconds. This test does not reproduce the
reference's within-world variation in DGPP. The campaign restores production
in finally and independently verifies binary/configuration identity and inference.
