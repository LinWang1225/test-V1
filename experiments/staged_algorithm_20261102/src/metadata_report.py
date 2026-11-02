"""Separate actual array inventory from proposed compact representations."""
from common import *
import numpy as np

def main():
    inventory=[]
    for codec in CODECS:
        files=sorted((ROOT/'codebooks'/codec).glob('layer*.npz'))
        totals={}
        for p in files:
            with np.load(p) as a:
                for k in a.files:
                    totals[k]=totals.get(k,0)+a[k].nbytes
        channels=len(files)*2*8*128
        curves={}
        for name,width,anchor in [('D_0_to_4',5,0),('gain_0_to_4',4,0),
            ('gain_0_to_4_with_D4',4,1),('D_2_to_4',3,0),('gain_2_to_4',2,0)]:
            curves[name]={f'fp{bits}_bytes':channels*(width+anchor)*(bits//8) for bits in (16,32,64)}
        inventory.append(dict(codec=codec,layers=len(files),actual_file_array_bytes=totals,
            actual_codebook_array_bytes=sum(v for k,v in totals.items() if k not in ('D','J')),
            proposed_curve_capacity=curves,
            local_FP16_scale_bits_per_KV_scalar_at_full_128_token_blocks=16/128,
            local_four_FP16_gains_bits_per_KV_scalar_at_full_128_token_blocks=4*16/128,
            local_peak_capacity_formula='2 sides * layers * 8 KV heads * ceil(T/128) blocks * 128 channels * 2 bytes * (1 scale or 4 gains)',
            scope='Uncompressed numeric-array capacity. File containers, reference expanded LUT and hardware traffic are separate.'))
    atomic(ROOT/'analysis/metadata_inventory.json',inventory)

if __name__=='__main__':
    main()
