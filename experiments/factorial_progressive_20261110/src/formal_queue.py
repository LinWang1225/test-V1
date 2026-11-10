import itertools,hashlib

def runs():
 primary=[{'run_id':'fp16','kind':'fp16','stage':'primary3','cost_mode':'payload'}]
 identities=[f'E{e}_B{b}_R{r}' for e,b,r in itertools.product(range(2),[1,0,2],range(4))]
 for ident in identities:
  for bit in [4,3]:primary.append({'run_id':f'{ident}_uniform{bit}','kind':'uniform','identity':ident,'uniform':bit,'budget':3,'cost_mode':'payload','stage':'primary3'})
  for c in range(4):primary.append({'run_id':f'{ident}_C{c}_payload3','kind':'dynamic','identity':ident,'controller':f'C{c}','budget':3,'cost_mode':'payload','stage':'primary3'})
 phases=[primary]
 for budget in [3,2]:
  if budget==2:
   phases.append([{'run_id':f'{ident}_uniform2','kind':'uniform','identity':ident,'uniform':2,'budget':2,'cost_mode':'payload','stage':'extension2'} for ident in identities]+[
    {'run_id':f'{ident}_C{c}_payload2','kind':'dynamic','identity':ident,'controller':f'C{c}','budget':2,'cost_mode':'payload','stage':'extension2'} for ident in identities for c in range(4)])
  phases.append([{'run_id':f'{ident}_C{c}_total{budget}','kind':'dynamic','identity':ident,'controller':f'C{c}','budget':budget,'cost_mode':'total','stage':f'total_cost{budget}'} for ident in identities for c in range(4)])
 assert len(primary)==145 and sum(len(x) for x in phases)==457
 return phases
