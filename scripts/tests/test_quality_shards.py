"""Check full coverage, reproducibility identity and conflicting-result refusal."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPTS=Path(__file__).resolve().parents[1]


def load(name):
    spec=importlib.util.spec_from_file_location(name,SCRIPTS/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class QualityShards(unittest.TestCase):
    def test_partition_is_disjoint_and_preserves_case_seeds(self):
        worker=load('run_quality_shard')
        rows=[dict(index=i,dataset='x',sampling_seed=42+i) for i in range(464)]
        groups=[worker.select_cases(rows,j,4)[1] for j in range(4)]
        flattened=[r for group in groups for r in group]
        self.assertEqual(len({r['index'] for r in flattened}),464)
        self.assertEqual(sorted(flattened,key=lambda r:r['index']),rows)
        with self.assertRaises(ValueError):worker.select_cases(rows,4,4)

    def fixture(self,root,index,text='same',fingerprint='fixed'):
        folder=root/'results/method';folder.mkdir(parents=True)
        manifest=dict(protocol_fingerprint=fingerprint,plan=dict(run_name='method',indices=[index]),
                      protocol=dict(method='method',eligible_indices=[0,1]))
        (root/'portable_manifest.json').write_text(json.dumps(manifest))
        (folder/f'{index:06d}.json').write_text(json.dumps(dict(index=index,run_fingerprint=fingerprint,text=text)))

    def test_merge_accepts_complementary_shards_and_rejects_conflicts(self):
        merge=load('merge_quality_shards')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=root/'a';b=root/'b';c=root/'c'
            self.fixture(a,0);self.fixture(b,1);self.fixture(c,0,text='different')
            _,records,_=merge.collect([a,b]);self.assertEqual(set(records),{0,1})
            with self.assertRaisesRegex(ValueError,'Conflicting'):merge.collect([a,c])

    def test_merge_rejects_different_protocols(self):
        merge=load('merge_quality_shards')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=root/'a';b=root/'b'
            self.fixture(a,0);self.fixture(b,1,fingerprint='other-method')
            with self.assertRaisesRegex(ValueError,'different methods'):merge.collect([a,b])


if __name__=='__main__':unittest.main()
