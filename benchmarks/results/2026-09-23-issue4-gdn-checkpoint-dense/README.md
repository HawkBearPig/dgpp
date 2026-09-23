# Chunked GDN with checkpoint dense weights — original failure persists

The unchanged original request still scores **5/6**, returning
`val_5dac9ed720abddaf` for the failed key, with 261120 prompt / 184 completion
tokens, zero cached tokens, HTTP 200 and stop, in 289.751 seconds. The request
SHA256 matches the original fixture. This combination is insufficient to repair
retrieval; the bounded follow-up was skipped.

The native 64-token block-GDN prototype is combined with checkpoint dense
projections. All other original request, MTP, KV and prefix-cache settings are
retained. Both rank logs verify checkpoint dense weights and no resident image
loads; the diagnostic getter disables both image reads and writes. The source
patch and exact server binary identity are recorded. Six operator tests were
executed in the preceding trace campaign, not rerun here; the kernel and test
sources are unchanged.

The first attempt stopped before inference because the harness expected the
peer startup log in the local collection directory. The retry fetches it from
the peer's resolved deployment directory before verification. The aborted
attempt is retained under `raw/attempt1-peer-log-guard` and excluded from
accuracy results. Both attempts restored and independently verified the exact
original four-node production binaries/configuration and inference smoke test.
