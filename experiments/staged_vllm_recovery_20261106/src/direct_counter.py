"""Single-pass CUPTI ranges; no Nsight launch tracing or generation replay."""
import csv
import ctypes
import time
import zlib
from pathlib import Path
import torch
from common import ROOT, atomic, sha

LIBRARY=ROOT/'native/libdirect_counter.so'


class DirectCounter:
    def __init__(self, base, execution_id):
        self.base=Path(base)
        self.base.parent.mkdir(parents=True,exist_ok=True)
        self.execution_id=execution_id
        self.lib=ctypes.CDLL(str(LIBRARY))
        self.lib.kv_error.restype=ctypes.c_char_p
        self.lib.kv_stop.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
        self.lib.kv_image_size.restype=ctypes.c_size_t
        self.lib.kv_image.restype=ctypes.c_void_p
        self.check(self.lib.kv_create())
        self.rows=[];self.active=None;self.overhead=0.0
        self.csv=self.base.with_suffix('.windows.csv')
        self.stream=self.csv.open('x',newline='')
        self.writer=csv.DictWriter(self.stream,fieldnames=['index','stage','units','dram_read_bytes','dram_write_bytes','dram_total_bytes','counter_seconds','raw_image','raw_image_sha256'])
        self.writer.writeheader();self.stream.flush()

    def check(self, code):
        if code:raise RuntimeError(self.lib.kv_error().decode())

    def start(self, stage):
        assert self.active is None and stage in ('prefill','decode')
        begin=time.perf_counter();torch.cuda.synchronize()
        self.check(self.lib.kv_start());self.active=stage
        self.overhead+=time.perf_counter()-begin

    def stop(self, units):
        assert self.active is not None and units>0
        begin=time.perf_counter();torch.cuda.synchronize()
        data=(ctypes.c_uint64*2)();self.check(self.lib.kv_stop(data))
        image=ctypes.string_at(self.lib.kv_image(),self.lib.kv_image_size())
        raw=self.base.parent/(self.base.name+f'.window{len(self.rows):05d}.counterdata.zlib')
        raw.write_bytes(zlib.compress(image,6))
        seconds=time.perf_counter()-begin;self.overhead+=seconds
        row=dict(index=len(self.rows),stage=self.active,units=units,
            dram_read_bytes=int(data[0]),dram_write_bytes=int(data[1]),dram_total_bytes=int(data[0]+data[1]),
            counter_seconds=seconds,raw_image=str(raw),raw_image_sha256=sha(raw))
        self.rows.append(row);self.writer.writerow(row);self.stream.flush();self.active=None
        return row

    def finish(self):
        assert self.active is None and self.rows
        self.check(self.lib.kv_close());self.stream.close()
        stages={}
        for stage in ('prefill','decode'):
            rows=[r for r in self.rows if r['stage']==stage]
            stages[stage]={k:sum(r[k] for r in rows) for k in ('dram_read_bytes','dram_write_bytes','dram_total_bytes','units')}
            stages[stage]['windows']=len(rows)
        total={k:sum(v[k] for v in stages.values()) for k in ('dram_read_bytes','dram_write_bytes','dram_total_bytes')}
        return dict(**total,stages=stages,profiled_ranges=len(self.rows),counter_collection_seconds=self.overhead,
            raw_csv=str(self.csv),raw_csv_sha256=sha(self.csv),counter_library_sha256=sha(LIBRARY),
            execution_id=self.execution_id,hardware_passes=1,
            measurement='direct CUPTI hardware DRAM byte counters, all consecutive live generation windows, no kernel trace or replay')
