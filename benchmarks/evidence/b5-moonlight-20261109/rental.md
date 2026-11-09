# Rental record, B5 report 1, 2026-11-09

Commands, in order. The instance API key that `vastai create` prints is not recorded.

```text
vastai create instance 27970173 --image nvidia/cuda:13.0.0-devel-ubuntu24.04 --disk 150 \
  --ssh --direct --label b5-03-moonlight --cancel-unavail --raw
    -> new_contract 59872202
vastai attach ssh 59872202 "$(cat ~/.ssh/id_ed25519.pub)"
scp -P 47746 repo.tgz remote-setup.sh remote-run.sh root@75.156.44.149:/root/b5/
ssh -f ... 'cd /root/b5 && nohup bash remote-setup.sh > /root/b5/setup.log 2>&1 < /dev/null &'
ssh -f ... 'cd /root/b5 && nohup bash remote-run.sh attempt1 > /root/b5/run-attempt1.out 2>&1 < /dev/null &'
echo y | vastai destroy instance 59872202
    -> "destroying instance 59872202."
vastai show instances --raw
    -> []          (saved in attempt1/show-instances-after.txt, taken 2026-11-09T10:22:28Z)
```

Code that ran on the host: `repo.tgz`, built with
`git archive HEAD src benchmarks/harness benchmarks/protocols/b5-moonlight-20261109 pyproject.toml`
at commit `5802c205093009d533786569f28704d06d4b8125`. SHA-256 of the tarball:
`4dbb763e0e1206106d0a314c332e12409641f5ca854fbf4e036e897711e04a47`. The journal shows
`soup-cli: null` because the code ran from `PYTHONPATH`, not from an installed package.

Cost. The hourly rate was 0.156 USD with the 150 GB disk, traffic 5.00 USD/TB. The instance
lived about 14 minutes (0.23 h): 0.04 USD, plus 32 GB of weights, 0.16 USD, about 0.20 USD in
all. Package downloads for the venv are not counted. The invoice of the day was not posted when
this was written. The team credit went from 124.46 to 124.18 USD, a drop of 0.28 USD that
includes other users of the shared account.
