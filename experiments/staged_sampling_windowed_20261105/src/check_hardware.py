"""Probe the same direct CUPTI backend required by actual requests."""
import torch
from common import *
from direct_counter import DirectCounter

def main():
    base=ROOT/'profiles/probes'/time.strftime('%Y%m%d_%H%M%S')
    try:
        torch.set_num_threads(4)
        x=torch.zeros(4<<20,device='cuda');torch.cuda.synchronize()
        counter=DirectCounter(base,'startup_probe')
        counter.start('decode');x.add_(1);counter.stop(1)
        measured=counter.finish()
        assert measured['dram_read_bytes']>0 and measured['dram_write_bytes']>0
        result=dict(available=True,attempted=True,probe_metrics=measured,backend='direct_cupti_windows')
    except Exception as exc:
        result=dict(available=False,attempted=True,reason=repr(exc),backend='direct_cupti_windows')
    result['checked_at']=time.strftime('%Y-%m-%d %H:%M:%S')
    atomic(ROOT/'hardware_capability.json',result)
    print('Direct CUPTI probe',result['available'],flush=True)
    if not result['available']:raise RuntimeError(result['reason'])

if __name__=='__main__':main()
