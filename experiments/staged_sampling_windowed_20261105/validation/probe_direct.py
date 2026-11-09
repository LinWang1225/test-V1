import sys,time,json,resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from direct_counter import DirectCounter

torch.set_num_threads(4)
x=torch.zeros(4<<20,device='cuda');torch.cuda.synchronize()
start=time.perf_counter();counter=DirectCounter(Path(sys.argv[1]),'hardware_probe')
for j in range(16):
    counter.start('decode')
    for i in range(16):x.add_(1)
    counter.stop(16)
receipt=counter.finish();torch.cuda.synchronize()
assert bool((x==256).all())
assert receipt['stages']['decode']['units']==256 and receipt['profiled_ranges']==16
assert receipt['dram_read_bytes']>0 and receipt['dram_write_bytes']>0
receipt.update(elapsed_seconds=time.perf_counter()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
Path(sys.argv[1]+'.validation.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))
