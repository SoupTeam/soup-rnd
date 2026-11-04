set -eu
date -u
cat /proc/sys/kernel/random/boot_id
nvidia-smi --query-gpu=uuid,name,driver_version,memory.total,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
findmnt -T /data -o TARGET,SOURCE,FSTYPE,OPTIONS
lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS,MODEL
stat -c '%u %g %a %F %n' /root /root/.ssh /root/.ssh/authorized_keys
python3 --version
df -B1 /data
cat /proc/driver/nvidia/version

python3 - <<'CHECK'
import sys,platform,venv,ensurepip,tarfile,os
assert sys.version_info >= (3,11)
assert platform.system() == 'Linux' and platform.machine() == 'x86_64'
assert platform.libc_ver()[0]=='glibc' and tuple(map(int,platform.libc_ver()[1].split('.'))) >= (2,39)
assert hasattr(tarfile,'data_filter')
assert not os.path.lexists('/data/c3')
print('PASS_OFFLINE_BOOTSTRAP_PREREQUISITES', sys.version, platform.libc_ver(), ensurepip.version())
CHECK

python3 - <<'RAM'
from pathlib import Path
import shutil
mem={r.split(':')[0]:int(r.split()[1])*1024 for r in Path('/proc/meminfo').read_text().splitlines() if r.startswith(('MemTotal:','MemAvailable:'))}
assert mem['MemAvailable']>=24*2**30,mem
assert shutil.disk_usage('/data').free>=65*2**30
print('MEMORY_AND_DISK',mem)
RAM

command -v curl
curl --version
python3 - <<'MEMORY'
from pathlib import Path
p=Path('/sys/fs/cgroup/memory.max')
if p.exists():
 value=p.read_text().strip()
 assert value=='max' or int(value)>=64_000_000_000, value
 print('CGROUP_MEMORY_MAX',value)
MEMORY
