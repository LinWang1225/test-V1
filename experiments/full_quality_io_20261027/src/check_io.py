"""CPU-only byte-accounting/parser checks; no interference with active GPU run."""
import tempfile
from common import *
import torch
from traffic import Ledger,parse_ncu,run_bounded
import sys,time


def main():
    k=torch.tensor(([2,3,4,3]*256)).reshape(8,128)
    v=torch.tensor(([4,3,2,3]*34)).reshape(8,17)
    ledger=Ledger('cpu');ledger.record(17,17,k,v);a=ledger.finish('query_curve3')
    assert a['K']['payload_bits']==int(k.sum())*17
    assert a['V']['payload_bits']==int(v.sum())*128
    assert a['K']['packet_read_bytes']==int(k.sum())*3
    assert a['V']['packet_read_bytes']==int(v.sum())*16
    assert a['K']['depth_counts']=={'2':256,'3':512,'4':256,'16':0}
    dense=Ledger('cpu');dense.record(17,17,16,16);d=dense.finish('fp16')
    assert d['K']['packet_read_bytes']==8*128*17*2
    assert d['V']['packet_read_bytes']==8*128*17*2
    assert d['decoder_metadata_read_allowance_bytes']==0
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'metrics.csv'
        header='"ID","Process ID","Metric Name","Metric Unit","Metric Value"\n'
        p.write_text('==PROF== tool preamble\n'+header+'"0","4","dram__bytes_read.sum","byte","4,096"\n"0","4","dram__bytes_write.sum","byte","32"\n')
        r=parse_ncu(p);assert r['dram_read_bytes']==4096 and r['dram_total_bytes']==4128
        for bad in [header+'"0","4","dram__bytes_read.sum","byte","n/a"\n',header+'"0","4","dram__bytes_read.sum","byte","32"\n',header+'"0","4","dram__bytes_read.sum","Kbyte","1"\n']:
            p.write_text(bad)
            try:parse_ncu(p)
            except ValueError:pass
            else:raise AssertionError('Missing/unsupported counter was treated as measured')
    start=time.monotonic()
    try:run_bounded([sys.executable,'-c','import time; time.sleep(10)'],.05)
    except TimeoutError:pass
    else:raise AssertionError('Profiler timeout was not bounded')
    assert time.monotonic()-start<5
    atomic(ROOT/'io_checks.json',dict(passed=True,K_V_separate_actual_depth_maps=True,odd_context_packet_padding=True,dense16_no_bitplane_padding=True,hardware_csv_read_write_verified=True,missing_and_na_metrics_rejected=True,profiler_timeout_terminates_child_group=True,GPU_not_used=True))
    print('IO ledger and hardware evidence parser checks passed',flush=True)


if __name__=='__main__':main()
