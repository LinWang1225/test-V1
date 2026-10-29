"""Attempt actual DRAM counters at each dataset boundary; never fail quality."""
import subprocess,time
from common import *
from traffic import NCU,parse_ncu,run_bounded


def main():
    stamp=time.strftime('%Y%m%d_%H%M%S');folder=ROOT/'profiles/probes';folder.mkdir(parents=True,exist_ok=True)
    raw=folder/(stamp+'.csv')
    command=[NCU,'--metrics','dram__bytes_read.sum,dram__bytes_write.sum','--nvtx','--nvtx-include','readkv_attention/','--cache-control','none','--clock-control','none','--csv','--page','raw','--print-units','base','--log-file',str(raw),str(ROOT.parents[1]/'.venv/bin/python'),str(ROOT/'src/counter_probe.py')]
    try:
        p=run_bounded(command,120)
        log=folder/(stamp+'.log');log.write_text(p.stdout)
        if p.returncode:raise RuntimeError(f'ncu exit {p.returncode}: '+p.stdout+(raw.read_text(errors='replace') if raw.exists() else ''))
        values=parse_ncu(raw)
        result=dict(available=True,attempted=True,checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),probe_metrics=values,command=command)
    except Exception as exc:
        result=dict(available=False,attempted=True,checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),reason=str(exc)[-6000:],command=command,run_without_hardware_measurement=True)
    atomic(ROOT/'hardware_capability.json',result)
    print('actual_hardware_counter_probe',result['available'],flush=True)


if __name__=='__main__':main()
