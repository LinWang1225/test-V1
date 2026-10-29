"""Frozen-trajectory single-layer experiment; no controller execution here."""
import argparse, traceback
from common import *


class EncodingExperiment(legacy.Experiment):
    def __init__(self, smoke=False):
        super().__init__(smoke)
        self.out = ROOT / 'results' / ('smoke' if smoke else 'full')
        self.out.mkdir(parents=True, exist_ok=True)
        self.newbooks = {(s, l): load_layer(l, s, smoke) for l in ((0,17,35) if smoke else range(36)) for s in (SEEDS[:1] if smoke else SEEDS)}

    def observe(self, module, q, k, v, scale):
        layer = module.layer_idx
        if self.smoke and layer not in (0,17,35):
            return
        Q = (q[0,:,0].float() @ self.U).view(8,4,128)
        K, V = k[0].float() @ self.U, v[0].float() @ self.U
        score = torch.einsum('hgd,htd->hgt', Q, K) * scale
        P = score.softmax(-1); O = P @ V
        core = dict(case=self.current['case_id'], dataset=self.current['dataset'], split='test', layer=layer, position=self.pos, context=K.shape[1])
        for seed in (SEEDS[:1] if self.smoke else SEEDS):
            book = self.newbooks[seed, layer]
            for method in METHODS:
                rec, _ = reconstruct(torch.stack((K,V)), method, book)
                dk = torch.stack([(rec[b][0]-K).double().square().mean(1) for b in BITS], -1)
                dv = torch.stack([(rec[b][1]-V).double().square().mean(2) for b in BITS], -1)
                for side, curves in [('K',dk), ('V',dv)]:
                    gains = curves[...,:-1]-curves[...,1:]
                    tol = 1e-10 + 1e-5 * curves.abs().amax(-1)
                    violation = (gains[...,1] > gains[...,0] + tol) | (gains.amin(-1) < -tol)
                    values = torch.cat((curves.mean(1), violation.double().mean(1)[:,None]), -1).cpu().numpy()
                    for h in range(8):
                        for j,b in enumerate(BITS):
                            norm = K[h].double().square().mean() if side == 'K' else V[h].double().square().mean()
                            self.distrows.append(dict(core, seed=seed, method=method, side=side, head=h, bits=b, mse=float(values[h,j]), normalized_mse=float(values[h,j]/max(float(norm),1e-20)), diminishing_violation_fraction=float(values[h,3])))
                scores = {b: torch.einsum('hgd,htd->hgt',Q,rec[b][0])*scale for b in BITS}
                probs = {b:s.softmax(-1) for b,s in scores.items()}
                for kb,vb in [(b,16) for b in BITS]+[(16,b) for b in BITS]+[(a,b) for a in BITS for b in BITS]:
                    s = score if kb==16 else scores[kb]
                    p = P if kb==16 else probs[kb]
                    vv = V if vb==16 else rec[vb][1]
                    out = p @ vv
                    # Accumulate error decompositions in float64 to retain cancellation.
                    delta = (vv-V).double()
                    value_error = p.double() @ delta
                    v_exact = value_error.square().mean((1,2))
                    v_diag = torch.einsum('hgt,ht->h',p.double().square(),delta.square().sum(-1))/(4*128)
                    key_error = (p.double()-P.double()) @ V.double()
                    interaction = 2*(key_error*value_error).mean((1,2))
                    measures = torch.stack(((out-O).square().mean((1,2)), (out-O).flatten(1).norm(dim=1)/O.flatten(1).norm(dim=1).clamp_min(1e-12), (s-score).square().mean((1,2)), (p-P).abs().sum(-1).mean(-1), v_diag, v_exact, v_exact-v_diag, key_error.square().mean((1,2)), interaction),-1).cpu().numpy()
                    for h,a in enumerate(measures):
                        self.rows.append(dict(core,seed=seed,method=method,head=h,k_bits=kb,v_bits=vb,**dict(zip(('output_mse','output_rel','score_mse','prob_l1','v_diag_mse','v_exact_mse','v_cross_mse','k_output_mse','kv_interaction_mse'),map(float,a)))))
        if layer==0 and self.pos==1:
            exact=(torch.einsum('hgd,htd->hgt',Q.double(),K.double())*scale).softmax(-1)@V.double()
            raw=(torch.einsum('hgd,htd->hgt',q[0,:,0].float().view(8,4,128),k[0].float())*scale).softmax(-1)@v[0].float()
            check=dict(fp32_fp64_rel=float((O.double()-exact).norm()/exact.norm()),rotation_rel=float((O@self.U.T-raw).norm()/raw.norm()))
            assert check['fp32_fp64_rel']<1e-4 and check['rotation_rel']<1e-3,check
            atomic(self.out/(self.current['case_id'].replace('/','_')+'_numerical.json'),check)

    def execute(self):
        selected=[r for r in self.manifest if r['split']=='test']
        for i,r in enumerate(selected):
            state('encoding_evaluate', completed=i,total=len(selected),case=r['case_id'],smoke=self.smoke)
            marker=self.out/(r['case_id'].replace('/','_')+'_complete.json')
            if marker.exists():
                trajectory_check(self.out,r,self.smoke)
                continue
            self.capture(r,'evaluate')
            trajectory_check(self.out,r,self.smoke)
            expected=(3 if self.smoke else 36)*4*(1 if self.smoke else 3)*len(METHODS)*8
            assert len(self.rows)==expected*15,(len(self.rows),expected*15)
            assert len(self.distrows)==expected*6
            atomic(marker,dict(complete=True,attention_rows=len(self.rows),distortion_rows=len(self.distrows)))
        atomic(self.out/'evaluation_complete.json',dict(complete=True,requests=len(selected),methods=METHODS,smoke=self.smoke))
        state('encoding_evaluation_completed',requests=len(selected),smoke=self.smoke)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    try:
        EncodingExperiment(args.smoke).execute()
    except BaseException:
        state('failed',traceback=traceback.format_exc());raise
