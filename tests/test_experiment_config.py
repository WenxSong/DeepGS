"""Contract checks for schema 2, independent of scientific model accuracy."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from deepgs.experiment import ExperimentConfig, load_experiment, validate_experiment


def minimal():
    return dict(schema_version=2, data=dict(path="input.npz"), preprocess=dict(differentiation="fd"), inversion=dict(enabled=False))


def test_yaml_paths_follow_config_not_working_directory(tmp_path, monkeypatch):
    cfgdir=tmp_path/'configs'
    cfgdir.mkdir()
    p=cfgdir/'run.yaml'
    p.write_text(yaml.safe_dump(minimal()),encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    cfg=load_experiment(p)
    assert cfg.data.path==str((cfgdir/'input.npz').resolve())
    assert cfg.inversion.enabled is False
    assert cfg.preprocess.field.dtype=='float64'


def test_dict_paths_use_cwd_and_input_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw=minimal()
    before=deepcopy(raw)
    cfg=load_experiment(raw)
    assert cfg.data.path==str(tmp_path/'input.npz')
    assert raw==before
    assert asdict(load_experiment(cfg.to_dict()))==asdict(cfg)


def test_spatial_selector_sets_explicit_cgsr_defaults():
    raw=minimal()
    raw.update(problem=dict(soil='heterogeneous'),structure=dict(selector='spatial_group'))
    cfg=load_experiment(raw)
    assert cfg.structure.grouping.normalization=='group'
    assert cfg.structure.sgtr.epsilon==pytest.approx(10**1.2)
    raw['structure']['sgtr']=dict(epsilon=12)
    assert load_experiment(raw).structure.sgtr.epsilon==12


@pytest.mark.parametrize('extra,match',[
    ({'pipeline':'gsr'},'Unknown option'),
    ({'preprocess':{'feild':{}}},'Unknown option'),
    ({'preprocess':{'seeds':[]}},'seeds'),
    ({'preprocess':{'seeds':[True]}},'expected'),
    ({'preprocess':{'field':{'dtype':'float16'}}},'dtype'),
    ({'preprocess':{'field':{'width':3.5}}},'expected'),
    ({'preprocess':{'field':{'learning_rate':float('nan')}}},'expected'),
    ({'problem':{'soil':'heterogeneous'}},'requires'),
    ({'structure':{'selector':'spline','grouping':{'normalization':'group'}}},'normalization'),
    ({'structure':{'grouping':{'by_depth':True}}},'Unknown option'),
    ({'inversion':{'coefficient_form':'bspline','network':{}}},'does not apply'),
    ({'inversion':{'coefficient_form':'polynomial','polynomial':{'signs':{'theta_z':0}}}},'signs'),
    ({'inversion':{'coefficient_form':'polynomial','polynomial':{'signs':{'madeup':1}}}},'signs'),
    ({'data':{'path':'a.npz','units':{'time':'fortnight'}}},'units.time'),
    ({'data':{'path':'a.npz','depth_positive':'up'}},'depth_positive'),
    ({'runtime':{'device':'cuda:banana'}},'runtime.device'),
])
def test_invalid_settings_fail_before_work(extra,match):
    raw=minimal()
    raw.update(extra)
    with pytest.raises(ValueError,match=match):
        load_experiment(raw)


def test_future_inversion_spec_is_accepted_but_not_executed():
    raw=minimal()
    raw['inversion']=dict(enabled=True,coefficient_form='bspline',bspline=dict(degree=3,n_internal_knots=6))
    cfg=load_experiment(raw)
    assert cfg.inversion.bspline.n_internal_knots==6


def test_mutated_dataclass_is_validated_as_strictly_as_yaml():
    cfg=load_experiment(minimal())
    cfg.preprocess.field.width=True
    with pytest.raises(ValueError,match='expected'):
        validate_experiment(cfg)


def test_csv_coordinate_units_are_explicit():
    raw=minimal()
    raw['data']=dict(path='sensor.csv',depths=[.1,.2],units=dict(time='hour'))
    with pytest.raises(ValueError,match='CSV coordinates'):
        load_experiment(raw)
    raw['data']['units']=dict(water_content='percent')
    assert load_experiment(raw).data.units.water_content=='percent'


def test_shipped_schema2_configs_parse():
    root=Path(__file__).resolve().parents[1]
    for p in (root/'configs').glob('selection_*_fd.yaml'):
        cfg=load_experiment(p)
        assert Path(cfg.data.path).is_absolute()
        assert not cfg.inversion.enabled


def test_missing_schema_and_wrong_container_are_rejected():
    with pytest.raises(ValueError,match='schema_version'):
        load_experiment(dict(data=dict(path='a.txt')))
    raw=minimal()
    raw['structure']=[]
    with pytest.raises(ValueError,match='mapping'):
        load_experiment(raw)
