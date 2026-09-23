# Second fresh Marlin trace — same wrong answer, differing expert values

The unchanged request again scores **5/6**, with the original wrong value
`val_5dac9ed720abddaf`, 261120 prompt / 184 completion tokens in 321.644 seconds.
Full assistant text and usage match the first trace exactly. The original
four-node production binaries/configuration and inference smoke test passed
after restoration.

On **both ranks**, all captured preceding layer-0 values agree, but its MoE
output differs in **14 of 5242880 elements**, relative L2 4.6013e-6 and maximum
absolute difference 0.00048828125. The first two chunks have full layer 0–3
captures, so this is a complete comparison at that first operator boundary.
The source overlays, checkpoint, inputs and launch settings are unchanged.
All recorded GDN launch configurations match at positions 0, 2048 and 260096.
Changing GDN launch choices therefore does not explain the first observed
variation in this pair of worlds.

Comparison covers 133678 common fields per rank with no input-token drift.
Rank 0 has 7126 equal / 126552 different fields; rank 1 has 7127 equal / 126551
different fields. The final trace call at position 261303 is incomplete in this
second capture (62 fields absent on each rank); the comparison reports that
boundary explicitly and does not count missing fields as equal. All saved
payloads with metadata pass length/SHA256 verification.

This localizes numerical variation to the reference MoE boundary. It does not
yet identify the operation inside that block or prove why the earlier Marlin
world succeeded: these two differing worlds still return the same failure.
A bounded first-chunk probe with internal expert captures is prepared.

As in the first trace, slow post-restoration recursive SCP was replaced with a
streamed tar transfer. The only campaign failure is the explicitly interrupted
copy after verified restoration; the successful replacement and payload
verification are recorded separately. GPU worlds remained serial.
