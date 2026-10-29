"""Scenario budget, replace rates with unprofiled pilot aggregate throughput."""
import argparse
p=argparse.ArgumentParser()
p.add_argument('--decode-tps', type=float, default=50)
p.add_argument('--prefill-tps', type=float, default=1000)
p.add_argument('--math-output', type=int, default=2048)
p.add_argument('--repeats', type=int, default=1)
a=p.parse_args()
if min(a.decode_tps,a.prefill_tps,a.math_output,a.repeats)<=0: p.error('values must be positive')
# Six LongBench categories, 50 stratified samples each; full HumanEval and MATH test.
inputs=300*8192+164*512+5000*512
outputs=300*256+164*512+5000*a.math_output
hours=2*a.repeats*(inputs/a.prefill_tps+outputs/a.decode_tps)/3600
print(f'FP16 + KVarN generation: {hours:.2f} hours; with 20% margin: {hours*1.2:.2f} hours')
print('Excludes install/build, scoring, profiling. Rates are assumptions unless supplied from pilot.')
