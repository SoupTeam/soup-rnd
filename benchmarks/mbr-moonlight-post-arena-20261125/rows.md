# Result lines (ticket 17, post-arena E, G and D+segmented)

- Moonlight-16B-A3B, RTX 3050 Laptop sm_86 4 GiB, window (8,64), arm C (ordered backward), mode speed (normal), torch 2.14.1+cu130, step 20.08 s, allocated 0.93 GiB, reserved 1.03 GiB, NVML 1.18 GiB, RAM 4.59 GiB, runs 10
- Moonlight-16B-A3B, RTX 3050 Laptop sm_86 4 GiB, window (8,64), arm E (C + lt-fused), mode speed (normal), torch 2.14.1+cu130, step 20.66 s, allocated 1.10 GiB, reserved 1.23 GiB, NVML 1.37 GiB, RAM 4.63 GiB, runs 10
- Moonlight-16B-A3B, RTX 3050 Laptop sm_86 4 GiB, window (8,64), arm G (E + async lease + fast row gather), mode speed (normal), torch 2.14.1+cu130, step 18.65 s, allocated 1.10 GiB, reserved 1.23 GiB, NVML 1.37 GiB, RAM 4.58 GiB, runs 10
- Moonlight-16B-A3B, RTX 3050 Laptop sm_86 4 GiB, window (8,512), arm C (ordered backward), mode speed (normal), torch 2.14.1+cu130, step 26.90 s, allocated 1.99 GiB, reserved 2.09 GiB, NVML 2.23 GiB, RAM 4.97 GiB, runs 10
- Moonlight-16B-A3B, RTX 3050 Laptop sm_86 4 GiB, window (8,512), arm E (C + lt-fused), mode speed (normal), torch 2.14.1+cu130, step 27.88 s, allocated 2.05 GiB, reserved 2.28 GiB, NVML 2.42 GiB, RAM 5.01 GiB, runs 10
- Moonlight-16B-A3B, RTX 3050 Laptop sm_86 4 GiB, window (8,512), arm G (E + async lease + fast row gather), mode speed (normal), torch 2.14.1+cu130, step 23.68 s, allocated 2.05 GiB, reserved 2.28 GiB, NVML 2.42 GiB, RAM 4.97 GiB, runs 10
- Moonlight-16B-A3B, RTX 3050 Laptop sm_86 4 GiB, window (1,64), arm E (C + lt-fused) against A0, mode bits (deterministic): equal, loss equal True, grads differing 0 / 216, max abs 0.00e+00, dataset_sha 6beb56ae
- Moonlight-16B-A3B, RTX 3050 Laptop sm_86 4 GiB, window (2,64), arm E (C + lt-fused) against A0, mode bits (deterministic): equal, loss equal True, grads differing 0 / 216, max abs 0.00e+00, dataset_sha 6beb56ae
- Moonlight-16B-A3B, RTX 3050 Laptop sm_86 4 GiB, window (4,64), arm E (C + lt-fused) against A0, mode bits (deterministic): equal, loss equal True, grads differing 0 / 216, max abs 0.00e+00, dataset_sha 6beb56ae
- Moonlight-16B-A3B, RTX 3050 Laptop sm_86 4 GiB, window (8,64), arm E (C + lt-fused) against A0, mode bits (deterministic): equal, loss equal True, grads differing 0 / 216, max abs 0.00e+00, dataset_sha 6beb56ae
- Moonlight-16B-A3B, RTX 3050 Laptop sm_86 4 GiB, window (1,512), arm E (C + lt-fused) against A0, mode bits (deterministic): equal, loss equal True, grads differing 0 / 216, max abs 0.00e+00, dataset_sha 91048d80
- Moonlight-16B-A3B, RTX 3050 Laptop sm_86 4 GiB, window (2,512), arm E (C + lt-fused) against A0, mode bits (deterministic): equal, loss equal True, grads differing 0 / 216, max abs 0.00e+00, dataset_sha 91048d80
- Moonlight-16B-A3B, RTX 3050 Laptop sm_86 4 GiB, window (4,512), arm E (C + lt-fused) against A0, mode bits (deterministic): equal, loss equal True, grads differing 0 / 216, max abs 0.00e+00, dataset_sha 91048d80
- Moonlight-16B-A3B, RTX 3050 Laptop sm_86 4 GiB, window (8,512), arm E (C + lt-fused) against A0, mode bits (deterministic): equal, loss equal True, grads differing 0 / 216, max abs 0.00e+00, dataset_sha 91048d80
- Moonlight-16B-A3B, RTX 3050 Laptop sm_86 4 GiB, window (2,64), arm G (E + async lease + fast row gather) against A0, mode bits (deterministic): equal, loss equal True, grads differing 0 / 216, max abs 0.00e+00, dataset_sha 6beb56ae
- Moonlight-16B-A3B, RTX 3050 Laptop sm_86 4 GiB, window (2,64), arm D (decoupled MBR) against A0, mode bits (deterministic): differs, loss equal False, grads differing 108 / 216, max abs 4.51e-02, dataset_sha 6beb56ae
- Moonlight-16B-A3B, RTX 3050 Laptop sm_86 4 GiB, window (2,64), arm DS (D + segmented) against A0, mode bits (deterministic): equal, loss equal True, grads differing 0 / 216, max abs 0.00e+00, dataset_sha 6beb56ae
