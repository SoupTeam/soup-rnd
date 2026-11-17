# D3 trained OLMoE attempt 01 — no verdict

Date: 2026-10-22. This is the retained first formal attempt for the trained-step study, not a passing result and not a timing sample.

- Frozen source commit: `a26f49484f88522ad7e34f6e3cb13b6893d45328`.
- Frozen protocol SHA-256: `1aaf1cd0f8d37a058d322c95c299a18351f25472b3623c31b74cab0bde1e8e46`.
- Session: `7ecbb5b9d3544bc0b9f7f68b041fc108`.
- Completed cell: `correctness-dense-native` (1 of the 84 planned cells).
- The full native dense OLMoE forward, backward and optimizer step ran; F1 returned `ok`. Peak GPU memory recorded by F1 was 15,630 MiB.
- The cell then failed while serializing tensor hashes: `tensor_hash` attempted a dtype view on the 0-D scalar loss and raised `RuntimeError: self.dim() cannot be 0`.
- Classification: `no_verdict`. The session stopped immediately. No timing cell ran, no result is included in the new series, and no replacement was made under protocol revision 1.

The archive contains the original session/result/log, exact protocol, frozen-source receipt and source packet. A corrected source and protocol revision 2 declare a separate fresh 84-cell series and explicitly supersede this retained attempt; they do not erase or relabel it.