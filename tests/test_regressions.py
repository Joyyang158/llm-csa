"""CPU regression coverage; no GPU libraries, model downloads, or API calls."""
import ast
import contextlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import pandas as pd

from src.csa.capability_ratio import accuracy_from_frame
from src.csa.evaluate import cds, macro_f1, report_metrics
from src.data.build_sft_dataset import build_sft_frame
from src.utils.data import science_choices, validate_query_alignment
from src.utils.grading import compute_is_correct, is_math_correct
from src.utils.parsing import extract_boxed_answer

ROOT = Path(__file__).resolve().parents[1]


def functions_from_source(module, names, **bindings):
    """Run the real function bodies with stubs for GPU-only dependencies."""
    tree = ast.parse((ROOT / module).read_text())
    nodes = ast.parse('from __future__ import annotations').body
    nodes += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {'os': os, **bindings}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), module, 'exec'), namespace)
    return namespace


def math_frame(samples):
    return pd.DataFrame({'question': ['q'], 'answer': ['1'], 'generation': [repr(samples)]})


def science_frame(correct):
    # Different shuffled gold letters test per-attempt remapping.
    attempts = [{'correct_label': letter, 'generation': f'\\boxed{{{letter if ok else "Z"}}}'}
                for letter, ok in zip('ABCDE', correct)]
    return pd.DataFrame({'question': ['q'], 'generation': [json.dumps(attempts)]})


class GradingTests(unittest.TestCase):
    def test_nested_box(self):
        text = r'Answer: \boxed{\frac{1}{2}}'
        self.assertEqual(extract_boxed_answer(text), r'\frac{1}{2}')
        self.assertEqual(is_math_correct(text, r'\frac{1}{2}'), 1)

    def test_escaped_braces_and_last_box(self):
        self.assertEqual(extract_boxed_answer(r'\boxed{0} then \boxed{\{1,2\}}'), r'\{1,2\}')
        self.assertEqual(extract_boxed_answer('<think>\\boxed{0}</think>\\boxed{\n1\n}').strip(), '1')

    def test_unclosed_final_box_is_invalid(self):
        self.assertIsNone(extract_boxed_answer(r'\boxed{1} then \boxed{\frac{1}{2}'))
        self.assertIsNone(extract_boxed_answer(None))

    def test_cr_math_degradation_does_not_use_any_correct(self):
        before = math_frame([r'\boxed{1}'] * 5)
        after = math_frame([r'\boxed{1}'] + [r'\boxed{0}'] * 4)
        self.assertEqual(compute_is_correct(before, 'generation', 'math').tolist(), [1])
        self.assertEqual(compute_is_correct(after, 'generation', 'math').tolist(), [1])
        pre = accuracy_from_frame(before, 'generation', 'math')[0]
        post = accuracy_from_frame(after, 'generation', 'math')[0]
        self.assertEqual(100 * post / pre, 20)

    def test_science_cr_and_majority_correct_are_distinct(self):
        frame = science_frame([1, 1, 0, 0, 0])
        self.assertEqual(accuracy_from_frame(frame, 'generation', 'science'), (0.4, 2, 5, 5))
        self.assertEqual(compute_is_correct(frame, 'generation', 'science').tolist(), [0])
        frame = science_frame([1, 1, 1, 0, 0])
        self.assertEqual(compute_is_correct(frame, 'generation', 'science').tolist(), [1])

    def test_incomplete_cr_attempts_rejected(self):
        a = math_frame([r'\boxed{1}'] * 5)
        b = math_frame([r'\boxed{1}'])
        for frame in [pd.concat([a, b]), math_frame([]), a.iloc[:0]]:
            with self.assertRaises(ValueError):
                accuracy_from_frame(frame, 'generation', 'math')

    def test_query_order_checked(self):
        a = pd.DataFrame({'question': ['first', 'second'], 'answer': ['1', '2']})
        validate_query_alignment(a, a.copy())
        with self.assertRaises(ValueError):
            validate_query_alignment(a, a.iloc[::-1])
        with self.assertRaises(ValueError):
            validate_query_alignment(a, a.iloc[:1])

    def test_cds_degenerate_cases(self):
        for values in [(1, 10, 0, 10), (0.5, 10, 0, 0), (0, 10, 0, 10)]:
            self.assertTrue(math.isnan(cds(*values)))
        self.assertAlmostEqual(cds(.85, 200, .2, 100), .65 / math.sqrt(.85*.15/200 + .2*.8/100))
        self.assertEqual(cds(.5, 10, .5, 10), 0)

    def test_empty_valid_predictions_report(self):
        frame = pd.DataFrame({'decision': pd.Series(dtype=int), 'is_correct': pd.Series(dtype=int)})
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            report_metrics(frame, 'decision', 2, 2, .5, [])
        self.assertIn('NA (empty group or zero standard error)', out.getvalue())
        self.assertIn('M-F1             : nan', out.getvalue())


class DataTests(unittest.TestCase):
    def setUp(self):
        self.model = pd.DataFrame({'question': ['a', 'b'], 'is_correct': [1, 0]})

    def test_self_sft_reordered(self):
        analyses = pd.DataFrame({'question': ['b', 'a'], 'is_correct': [0, 1],
                                 'routing_analysis': ['cannot', 'can']})
        result = build_sft_frame(self.model, analyses, 'self')
        self.assertEqual(result.SFT_analysis.tolist(), ['can', 'cannot'])

    def test_teacher_sft_selects_matching_label(self):
        analyses = pd.DataFrame({'question': ['a', 'b'], 'label_0_analysis': ['no-a', 'no-b'],
                                 'label_1_analysis': ['yes-a', 'yes-b']})
        self.assertEqual(build_sft_frame(self.model, analyses).SFT_analysis.tolist(), ['yes-a', 'no-b'])

    def test_self_sft_rejects_mismatched_conditioning(self):
        analyses = self.model.assign(routing_analysis=['can', 'cannot'])
        analyses['is_correct'] = [0, 1]
        with self.assertRaisesRegex(ValueError, 'labels differ'):
            build_sft_frame(self.model, analyses, 'self')

    def test_sft_rejects_missing_duplicate_or_failed_analyses(self):
        analyses = self.model.assign(routing_analysis=['can', 'cannot'])
        for bad in [analyses.iloc[:1], pd.concat([analyses, analyses]),
                    analyses.assign(routing_analysis=['Error: Timeout', 'cannot'])]:
            with self.assertRaises(ValueError):
                build_sft_frame(self.model, bad, 'self')

    def test_science_query_formats(self):
        expected = '(A) first\n(B) second'
        for value in [['first', 'second'], "['first', 'second']"]:
            self.assertEqual(science_choices({'options': value}), expected)
        self.assertEqual(science_choices({'choices': expected}), expected)
        row = pd.read_csv(ROOT / 'dataset/science/train.csv').iloc[0].to_dict()
        fn = functions_from_source('src/training/grpo.py', ['build_query'], science_choices=science_choices)
        self.assertIn('(A)', fn['build_query'](row, 'science'))
        fn = functions_from_source('src/training/sft.py', ['_user_content'], science_choices=science_choices)
        self.assertIn('(A)', fn['_user_content']('{question}\n{choices}', row, 'science'))


class ConfigTests(unittest.TestCase):
    def test_training_config_forwards_length_and_seed(self):
        for path, ctor, expected in [('src/training/sft.py', 'SFTConfig', {'max_length': 5000, 'seed': 3407}),
                                     ('src/training/grpo.py', 'GRPOConfig', {'seed': 3407})]:
            tree = ast.parse((ROOT / path).read_text())
            fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_training_args')
            keys = [n.slice.value for n in ast.walk(fn) if isinstance(n, ast.Subscript)
                    and isinstance(n.value, ast.Name) and n.value.id == 'config' and isinstance(n.slice, ast.Constant)]
            config = dict.fromkeys(keys, 1)
            config.update(base_output_dir='out', max_seq_length=5000, seed=3407)
            ns = functions_from_source(path, ['build_training_args'], **{ctor: lambda **kwargs: kwargs})
            args = ns['build_training_args'](config)
            for key, value in expected.items():
                self.assertEqual(args[key], value)

    def test_single_inference_forwards_sampling(self):
        generate = Mock(return_value=('<decision>DELEGATE</decision>',))
        generate.return_value = (['<decision>DELEGATE</decision>'], .1)
        ns = functions_from_source('src/csa/inference_local.py', ['run_single'],
            get_prompt_template=lambda **kw: '{query}', build_query=lambda row, domain: row['question'],
            tqdm=lambda it, **kw: it, logging=Mock(), generate_vllm=generate,
            parse_decision=lambda text: 0, save_checkpoint=Mock(), np=SimpleNamespace(mean=lambda a: sum(a)/len(a)))
        with tempfile.TemporaryDirectory() as d:
            args = SimpleNamespace(binary_only=False, domain='math', model_type='qwen', max_new_tokens=20,
                enable_thinking=False, temperature=.2, top_p=.8, top_k=7, warmup=0,
                decision_col='decision', analysis_col='analysis', output_csv=f'{d}/out.csv', save_every=10)
            ns['run_single'](pd.DataFrame({'question': ['q']}), 0, object(), object(), args)
        kwargs = generate.call_args.kwargs
        self.assertEqual((kwargs['temperature'], kwargs['top_p'], kwargs['top_k']), (.2, .8, 7))


class LauncherTests(unittest.TestCase):
    def call_launcher(self, path, **env):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            python = root / 'python'
            python.write_text(f'#!{sys.executable}\nimport json, os, sys\nwith open(os.environ["CAPTURE"], "a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n')
            python.chmod(0o755)
            process = subprocess.run(['bash', str(ROOT/path)], cwd=ROOT,
                env={**os.environ, 'PATH': f'{d}:'+os.environ['PATH'], 'CAPTURE': f'{d}/args.json', **env},
                capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            return [json.loads(line) for line in (root/'args.json').read_text().splitlines()]

    def test_generation_honors_checkpoint_and_output(self):
        calls = self.call_launcher('scripts/data/generate_answers.sh', MODEL_NAME='/models/trained checkpoint',
                                   OUTPUT_CSV='/tmp/answers with spaces.csv', DOMAIN='science', SEED='123')
        self.assertEqual(len(calls), 1)
        args = calls[0]
        self.assertEqual(args[args.index('--model_name')+1], '/models/trained checkpoint')
        self.assertEqual(args[args.index('--output_csv')+1], '/tmp/answers with spaces.csv')
        self.assertEqual(args[args.index('--output_col')+1], 'generation')
        self.assertEqual(args[args.index('--seed')+1], '123')
        self.assertIn('--num_shuffles', args)

    def test_paths_connect_generation_grading_dfw(self):
        env = dict(MODEL_NAME='Qwen/Qwen3-4B', MODEL_TAG='my-base', DOMAIN='science', SPLIT='train')
        gen = self.call_launcher('scripts/data/generate_answers.sh', **env)[0]
        grade = self.call_launcher('scripts/evaluation/grade_science.sh', **env)[0]
        dfw = self.call_launcher('scripts/training/run_dfw.sh', **env)[0]
        self.assertEqual(gen[gen.index('--output_csv')+1], grade[grade.index('--csv_path')+1])
        self.assertEqual(grade[grade.index('--output_path')+1], dfw[dfw.index('--input_csv')+1])

    def test_builder_keeps_explicit_paths(self):
        args = self.call_launcher('scripts/data/build_sft_dataset.sh', MODEL_CSV='/tmp/model.csv',
                                  ANALYSIS_CSV='/tmp/self.csv', OUTPUT_CSV='/tmp/result.csv', SFT_MODE='self')[0]
        for flag, value in [('--model_csv', '/tmp/model.csv'), ('--analysis_csv', '/tmp/self.csv'),
                            ('--output_csv', '/tmp/result.csv'), ('--mode', 'self')]:
            self.assertEqual(args[args.index(flag)+1], value)

    def test_evaluation_defaults_to_fresh_generations(self):
        args = self.call_launcher('scripts/evaluation/evaluate_csa.sh', CSV_PATH='pred.csv', GT_CSV_PATH='post.csv')[0]
        self.assertEqual(args[args.index('--grade_mode')+1], 'generations')

    def test_local_inference_uses_released_dataset(self):
        args = self.call_launcher('scripts/inference/inference_local.sh', DOMAIN='science', MODEL_NAME='/models/checkpoint')[0]
        self.assertEqual(args[args.index('--input_csv')+1], 'dataset/science/test.csv')

    def test_all_shell_syntax(self):
        for path in (ROOT/'scripts').rglob('*.sh'):
            result = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


class CliTests(unittest.TestCase):
    def run_cli(self, module, *args):
        return subprocess.run([sys.executable, '-m', module, *map(str, args)], cwd=ROOT,
                              capture_output=True, text=True)

    def test_grade_cr_and_csa_pipeline(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            for domain in ['math', 'science']:
                before = math_frame([r'\boxed{1}']*5) if domain == 'math' else science_frame([1]*5)
                after = math_frame([r'\boxed{1}']+[r'\boxed{0}']*4) if domain == 'math' else science_frame([1, 0, 0, 0, 0])
                before.to_csv(p/'pre.csv', index=False)
                after.to_csv(p/'post.csv', index=False)
                grade = self.run_cli('src.data.grade_answers', '--csv_path', p/'post.csv',
                    '--eval_col', 'generation', '--domain', domain, '--output_path', p/'graded.csv')
                self.assertEqual(grade.returncode, 0, grade.stderr)
                expected_label = 1 if domain == 'math' else 0
                self.assertEqual(pd.read_csv(p/'graded.csv').is_correct.tolist(), [expected_label])
                cr = self.run_cli('src.csa.capability_ratio', '--pre_csv', p/'pre.csv',
                    '--post_csv', p/'post.csv', '--domain', domain)
                self.assertEqual(cr.returncode, 0, cr.stderr)
                self.assertIn('CR                    : 20.0%', cr.stdout)
                after[['question']].assign(decision=[expected_label]).to_csv(p/'pred.csv', index=False)
                evaluation = self.run_cli('src.csa.evaluate', '--csv_path', p/'pred.csv',
                    '--prediction_col', 'decision', '--grade_mode', 'generations',
                    '--gt_csv_path', p/'post.csv', '--generations_col', 'generation', '--domain', domain)
                self.assertEqual(evaluation.returncode, 0, evaluation.stderr)
                self.assertIn('Accuracy         : 1.000', evaluation.stdout)
                self.assertIn('CDS              : NA', evaluation.stdout)

    def test_cr_rejects_mismatched_attempt_counts(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            math_frame([r'\boxed{1}']*5).to_csv(p/'pre.csv', index=False)
            math_frame([r'\boxed{1}']).to_csv(p/'post.csv', index=False)
            result = self.run_cli('src.csa.capability_ratio', '--pre_csv', p/'pre.csv',
                '--post_csv', p/'post.csv', '--domain', 'math')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('same number of attempts', result.stderr)

    def test_evaluation_rejects_unknown_predictions_and_labels(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'input.csv'
            for decision, label in [(float('nan'), 1), (7, 1), (1, .5)]:
                pd.DataFrame({'decision': [decision], 'is_correct': [label]}).to_csv(path, index=False)
                result = self.run_cli('src.csa.evaluate', '--csv_path', path,
                    '--prediction_col', 'decision', '--grade_mode', 'column')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('ValueError', result.stderr)

    def test_self_sft_cli(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            labels = pd.DataFrame({'question': ['q'], 'is_correct': [1]})
            labels.to_csv(p/'labels.csv', index=False)
            labels.assign(routing_analysis=['I can solve it.']).to_csv(p/'analysis.csv', index=False)
            result = self.run_cli('src.data.build_sft_dataset', '--model_csv', p/'labels.csv',
                '--analysis_csv', p/'analysis.csv', '--output_csv', p/'sft.csv', '--mode', 'self')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(pd.read_csv(p/'sft.csv').SFT_analysis.tolist(), ['I can solve it.'])


if __name__ == '__main__':
    unittest.main()
