Host check for the required `foreign_reads` validity check (2026-10-29), done after the PR #46 review.

- Ordinary vast.ai instance (RTX 3060, Poland, machine 19544): `/root` is on overlay, `disk_device("/root")`
  is `None`, so `foreign_reads` is `unknown`. Same for every earlier rented host. Files bind-mounted from the
  host (`/etc/hosts`) do resolve to `nvme0n1`, but the model cache is not one of them.
- Instance with a vast local volume (`--create-volume`, `--mount-path /root/volume`, RTX 5070 Ti, Texas,
  machine 62158, EPYC 7V13, volume xfs on `/dev/nvme0n1p1`): `disk_device("/root/volume")` is `nvme0n1`.
  `ArmWatch` over 15 s with `model_path=/root/volume`: outcome `ok`, suspend `ok`, power `ok`,
  foreign_reads `ok` ("at most 0 bytes of foreign reads on nvme0n1").
- So the check answers when the weights and the cache sit on a local volume. Only one machine in the
  offers had both a rentable GPU and a volume ask; its card is a 16 GB RTX 5070 Ti at about 0.80 USD/h.
- Instance and volume destroyed afterwards (`vastai show volumes` lists none of mine).
