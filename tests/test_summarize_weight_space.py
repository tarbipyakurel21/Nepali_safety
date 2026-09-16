import json
import tempfile
import unittest
from pathlib import Path
from src.summarize_weight_space import summarize


class SummaryTests(unittest.TestCase):
    def test_paired_counts_and_incomplete_verdict_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/'data';data.mkdir();(data/'manifest.json').write_text(json.dumps({'eval_indices':[7,9]}))
            run=root/'run';run.mkdir()
            config={'complete':True,'model':'test','revision':'fixed','max_new_tokens':128,
                    'eval_sha256':'same','belebele_sha256':'same','comprehension_ids_hash':'same'}
            for c in ['base','random','attack']:
                out=run/c;(out/'verdicts').mkdir(parents=True)
                (out/'evaluation.json').write_text(json.dumps(config))
                (out/'comprehension.json').write_text(json.dumps([{'id':'one','language':lang,'gold':'A','prediction':'A','correct':True} for lang in ['eng_Latn','npi_Deva']]))
                for stem in ['english','nepali','romanized']:
                    labels=[{'global_index':7,'label':'unsafe' if c=='attack' else 'safe'},{'global_index':9,'label':'invalid'}]
                    (out/'verdicts'/f'insecure_llama_guard_{stem}.json').write_text(json.dumps(labels))
            summarize(run,data);r=json.loads((run/'summary.json').read_text())
            self.assertEqual(r['english/attack_vs_base']['delta_pp'],50.)
            self.assertEqual(r['attack/english']['invalid'],1)
            self.assertEqual(r['english/attack_vs_base']['transitions']['safe_to_unsafe'],1)
            (run/'attack/verdicts/insecure_llama_guard_english.json').write_text('[]')
            with self.assertRaises(ValueError):summarize(run,data)

if __name__=='__main__':unittest.main()
