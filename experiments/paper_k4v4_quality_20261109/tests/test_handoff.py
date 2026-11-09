"""Verify cancellation bookkeeping and service ordering without admin or GPU."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]


class HandoffTest(unittest.TestCase):
    def test_complete_gaussian_handoff_keeps_u_clipped_unattempted(self):
        spec=importlib.util.spec_from_file_location('checked_handoff',ROOT/'ops/handoff.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            project=Path(directory);root=project/'next';parent=project/'parent'
            root.mkdir();parent.mkdir();(project/'locks').mkdir()
            (project/'locks/staged_algorithm_gpu.lock').touch()
            folder=parent/'results/Gaussian_Tree__fixed4';(folder/'scores').mkdir(parents=True)
            for i in range(464):
                (folder/f'{i:06d}.json').write_text('{}')
                (folder/'scores'/f'{i:06d}.json').write_text('{}')
            (parent/'stage_status.json').write_text(json.dumps({'stage':'codec_complete'}))
            module.ROOT=root;module.PROJECT=project;module.PARENT=parent
            with patch.object(module.os,'geteuid',return_value=0),patch.object(module.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as run:
                module.main()
            calls=[call.args[0] for call in run.call_args_list]
            self.assertEqual(calls[0],['systemctl','stop',module.UNIT])
            self.assertEqual(calls[1],['systemctl','disable',module.UNIT])
            status=json.loads((root/'handoff_status.json').read_text())
            self.assertEqual(status['stage'],'ready_for_paper_methods')
            self.assertEqual(status['u_clipped_generated'],0)
            self.assertEqual(json.loads((parent/'stage_status.json').read_text())['skipped_methods'],['U_Clipped'])
            self.assertFalse((root/'references').exists())
            with patch.object(module.os,'geteuid',return_value=0),patch.object(module.subprocess,'run') as repeated:
                module.main()
                repeated.assert_not_called()


if __name__=='__main__':unittest.main()
