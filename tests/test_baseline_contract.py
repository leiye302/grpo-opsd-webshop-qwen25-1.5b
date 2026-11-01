import ast
import hashlib
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import UPSTREAM,load
from run import config


def test_public_baseline_anchor_and_named_beta_override(tmp_path):
    lock=load('upstream.json')
    assert lock['native_launcher']=='examples/grpo_opsd_trainer/run_webshop_3b.sh'
    text=(UPSTREAM/lock['native_launcher']).read_text(encoding='utf-8')
    assert re.search(r'(?m)^gate_beta=0[.]0$',text)
    assert re.search(r'(?m)^sdar_coef=0[.]01$',text)
    cfg=config('1.5b',tmp_path/'base',tmp_path/'run')
    assert cfg.algorithm.sdar.gate_beta==5.0
    assert cfg.algorithm.sdar.sdar_coef==.01
    assert cfg.actor_rollout_ref.actor.get('pg_loss_coef',1.)==1.
    assert cfg.data.val_batch_size==140 and cfg.trainer.total_training_steps==150
    assert 'local_hints' not in cfg.algorithm and cfg.get('opd_teacher') is None
    assert lock['baseline_variant']['teacher']=='pre_update_current_actor'
    assert lock['baseline_variant']['local_hint_api'] is False
    assert lock['baseline_variant']['rethink'] is False
    assert cfg.ray_init.runtime_env.env_vars.OPSD_RUN_DIR==str(tmp_path/'run')


def test_beta5_really_restores_teacher_dependent_student_gradient():
    torch=pytest.importorskip('torch')
    path=UPSTREAM/'verl/trainer/ppo/sdar_utils.py'
    node=next(n for n in ast.parse(path.read_text()).body
              if isinstance(n,ast.FunctionDef) and n.name=='compute_sdar_loss')
    scope={'torch':torch,'agg_loss':lambda loss_mat,loss_mask,loss_agg_mode:
           (loss_mat*loss_mask).sum()/loss_mask.sum()}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    def gradient(beta,teacher_value):
        student=torch.tensor([[-1.]],requires_grad=True)
        teacher=torch.tensor([[teacher_value]],requires_grad=True)
        loss,_=scope['compute_sdar_loss'](student,teacher,torch.ones_like(student),gate_beta=beta)
        loss.backward()
        assert teacher.grad is None
        return student.grad.item()
    assert gradient(0.,-.1)==gradient(0.,-3.)==-.5
    assert abs(gradient(5.,-.1)-gradient(5.,-3.))>.9
