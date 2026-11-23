# Ticket 13: memory rows, C and E, normal mode

One line per measured cell; two passes were run and agree byte for byte (the second is the file
`rows_canonical.jsonl`; the first is `raw/first-pass-rows/`). Every line names model, card,
window, arm, mode and torch version.

- Moonlight-16B-A3B revision `2a9a177f4841649f849ba941e41d2ee73c042314`, NVIDIA GeForce RTX 3050 Laptop GPU, window (8,64), arm C, mode normal, torch 2.14.1+cu130: ok, allocated 954.6 MiB, reserved 1146.0 MiB, NVML 1292.0 MiB, RAM 4682.4 MiB, AC online 1.
- Moonlight-16B-A3B revision `2a9a177f4841649f849ba941e41d2ee73c042314`, NVIDIA GeForce RTX 3050 Laptop GPU, window (8,64), arm E, mode normal, torch 2.14.1+cu130: ok, allocated 994.6 MiB, reserved 1144.0 MiB, NVML 1290.0 MiB, RAM 4719.1 MiB, AC online 1.
- Moonlight-16B-A3B revision `2a9a177f4841649f849ba941e41d2ee73c042314`, NVIDIA GeForce RTX 3050 Laptop GPU, window (8,512), arm C, mode normal, torch 2.14.1+cu130: ok, allocated 2066.2 MiB, reserved 2158.0 MiB, NVML 2304.0 MiB, RAM 5063.8 MiB, AC online 1.
- Moonlight-16B-A3B revision `2a9a177f4841649f849ba941e41d2ee73c042314`, NVIDIA GeForce RTX 3050 Laptop GPU, window (8,512), arm E, mode normal, torch 2.14.1+cu130: ok, allocated 2204.0 MiB, reserved 3310.0 MiB, NVML 3458.0 MiB, RAM 5112.2 MiB, AC online 1.
- Moonlight-16B-A3B revision `2a9a177f4841649f849ba941e41d2ee73c042314`, NVIDIA GeForce RTX 3050 Laptop GPU, window (64,64), arm C, mode normal, torch 2.14.1+cu130: ok, allocated 1725.8 MiB, reserved 1932.0 MiB, NVML 2078.0 MiB, RAM 5086.0 MiB, AC online 1.
- Moonlight-16B-A3B revision `2a9a177f4841649f849ba941e41d2ee73c042314`, NVIDIA GeForce RTX 3050 Laptop GPU, window (64,64), arm E, mode normal, torch 2.14.1+cu130: ok, allocated 1865.3 MiB, reserved 2102.0 MiB, NVML 2250.0 MiB, RAM 5130.0 MiB, AC online 1.
