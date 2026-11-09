# Rental record, B5 report 3, 2026-11-09

The instance API key that `vastai create` prints is not recorded. Times are UTC.

```text
Attempt 1  vastai create instance 59868928 --image nvidia/cuda:13.0.0-devel-ubuntu24.04 --disk 500 \
             --ssh --direct --label b5-06-synth671b --cancel-unavail --raw
             -> new_contract 59900998 (VM offer, machine 56506, Quebec, RTX 3070, 0.198 USD/h at 500 GB)
             created 14:42:00. ssh refused: sshd log "bad ownership or modes for file
             /root/.ssh/authorized_keys". Destroyed after about 6 minutes. No run.
Attempt 2  vastai create instance 51184871 ... --disk 480 --label b5-06-synth671b-2
             -> new_contract 59901632 (container, machine 12767, Sweden, RTX 3060 Ti)
             created 14:48:27, destroyed on the user's request before the first login
             (about 1 minute). No run.
Attempt 3  vastai create instance 58836814 --image nvidia/cuda:13.0.0-devel-ubuntu24.04 --disk 480 \
             --ssh --direct --label b5-06-synth671b-3 --cancel-unavail --raw
             -> new_contract 59901752 (container, machine 54551, Quebec, RTX 3060 Ti,
                0.267 USD/h at 480 GB: GPU 0.133 + disk 0.133)
             created 14:49:52
vastai attach ssh 59901752 "$(cat ~/.ssh/id_ed25519.pub)"
scp -P 53071 repo.tgz remote-setup.sh remote-write.sh remote-run.sh root@<host-ip>:/root/b5/
ssh -f ... 'cd /root/b5 && nohup bash -c "bash remote-setup.sh && bash remote-write.sh store" &'
  16:05  the host stopped the instance (intended_status "stopped") with 94 of 122 store files done.
         I did not stop it. The vast account is shared, so a team member cannot be excluded.
  16:10  vastai start instance 59901752 -> running again, disk intact
scp repo2.tgz remote-resume.sh   (code of commit 20c931d8: writer --resume)
ssh -f ... 'bash remote-resume.sh'      # store write finished 16:36:51
ssh -f ... 'bash remote-run.sh cold1 --cold'   # 16:37 to 16:45
ssh -f ... 'bash remote-run.sh warm1 --warm'   # 16:46 to 16:50
echo y | vastai destroy instance 59901752      # 16:51:37
vastai show instances --raw   -> only an instance that is not ours (RTX 3050, 59875508)
                                 (attempt3/show-instances-after.json)
```

Code that ran: `repo.tgz` of commit `39b7c6f3c3d1bc45fd624543ed6f318a2e12a27e` (sha256 in
`attempt3/repo-tgz.sha256`) and, from the resume on, `repo2.tgz` of commit
`20c931d8c195b7976849e3b07783006c82c3bbd5` (`attempt3/repo2-tgz.sha256`). The journals show
`soup-cli: null` because the code ran from `PYTHONPATH`.

Cost, estimated (the invoice of the day was not posted when this was written):

| Part | Basis | USD |
| --- | --- | --- |
| attempt 3, instance | 7304 s (2.03 h) x 0.2667 USD/h, less about 5 minutes of GPU (0.133 USD/h) while stopped, the disk is billed while stopped | 0.53 |
| attempt 3, traffic | `inet_down_billed` 3 327 887 read as kilobytes (3.33 GB, the unit is assumed) x 20 USD/TB (venv packages) | 0.07 |
| attempt 1 | about 6 minutes x 0.198 USD/h | 0.02 |
| attempt 2 | about 1 minute x 0.231 USD/h | 0.004 |
| total | | about 0.62 |

Team credit: 118.743 USD before (14:42), 117.188 USD after (16:52). The drop of 1.556 USD
is about 0.94 USD more than my estimate. A possible cause is the other instance on the shared
account, which ran the whole time; that is not verified. The ceiling the user confirmed is 3 USD.
