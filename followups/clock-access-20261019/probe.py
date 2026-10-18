import json
import time

import cloud_probe as cloud

lease=json.loads((cloud.ROOT/'lease.json').read_text())
deadline=lease['created_unix']+300
try:
    while time.time()<deadline:
        if (cloud.ROOT/'released.json').exists():raise RuntimeError('guard already released lease')
        row=cloud.instance(lease['instance_id'])
        if not row:raise RuntimeError('lease absent')
        assert row['label']==cloud.LABEL
        cloud.save('startup-status.json',{'status':row.get('actual_status'),'elapsed_s':time.time()-lease['created_unix']})
        if row.get('actual_status')!='running':
            time.sleep(10)
            continue
        client=None
        try:
            client=cloud.connect(row)
            script=r'''import json,subprocess,re,pathlib
def run(*args):
 p=subprocess.run(['nvidia-smi',*args],capture_output=True,text=True,timeout=20)
 return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
query=run('--query-gpu=uuid,name,driver_version,power.limit','--format=csv,noheader')
lines=query['stdout'].strip().splitlines()
assert len(lines)==1 and ', NVIDIA GeForce RTX 3090,' in lines[0]
uuid=lines[0].split(',')[0].strip()
assert re.fullmatch(r'GPU-[a-fA-F0-9-]+',uuid)
pathlib.Path('/workspace/d1probe/clock_uuid').write_text(uuid)
result={'identity':query,'lock':None,'reset':None,'scope':'clock-access-only; no F1 executed'}
try:
 result['lock']=run('-i',uuid,'-lgc','1200,1200')
 result['clock_settings']=run('-i',uuid,'-q','-d','CLOCK')
finally:result['reset']=run('-i',uuid,'-rgc')
print(json.dumps(result))
'''
            _,out,err=client.exec_command("python - <<'PY'\n"+script+'\nPY',timeout=60)
            data=out.read().decode()
            error=err.read().decode()
            assert out.channel.recv_exit_status()==0,error
            result=json.loads(data)
            result.update(instance_id=lease['instance_id'],checked_unix=time.time())
            cloud.save('clock-access.json',result)
            print(json.dumps(result),flush=True)
            break
        except Exception as error:
            cloud.save('probe-error.json',{'type':type(error).__name__,'message':str(error),'time':time.time()})
            time.sleep(5)
        finally:
            if client:client.close()
    else:
        cloud.save('startup-timeout.json',{'max_seconds':300,'cuda_run':False,'f1_run':False})
        print('Startup timeout; no CUDA or F1 executed',flush=True)
finally:
    cloud.destroy(lease)
    print('Exact probe lease deleted and absence verified',flush=True)
