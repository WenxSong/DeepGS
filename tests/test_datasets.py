"""Explicit-coordinate public data contract, without model training."""
from pathlib import Path
import json

import numpy as np
import pytest
import yaml

from deepgs.cli import main
from deepgs.data import Observations
from deepgs.datasets import (load_standard_csv, load_standard_npz, validate_dataset,
                             validate_observations)
from deepgs.experiment import load_experiment
from deepgs.workflow import _dataset_provenance, load_data


def arrays():
    return dict(times=np.array([0.,.0008,.015,.13]),depths=np.array([.05,.10,.25]),
                theta=np.array([[.2,.21,.23],[.22,.23,.25],[.24,.25,.27],[.28,.29,.31]]))


def write_csv(path,values):
    header='time,'+','.join(f'{d:g}m' for d in values['depths'])
    np.savetxt(path,np.column_stack((values['times'],values['theta'])),delimiter=',',header=header,comments='')


def metadata(observations='observations.npz'):
    return dict(schema_version=1,name='format-demo',kind='synthetic',observations=observations,
                units=dict(time='day',depth='m',water_content='fraction'),depth_positive='down',
                time_origin='start_of_simulation',source='Generated only for format tests',license='CC0-1.0')


def write_dataset(folder,meta=None,values=None):
    folder.mkdir(exist_ok=True)
    meta=metadata() if meta is None else meta
    (folder/'metadata.yaml').write_text(yaml.safe_dump(meta),encoding='utf-8')
    if meta['observations']=='observations.npz':
        np.savez(folder/'observations.npz',**(arrays() if values is None else values))
    elif meta['observations']=='observations.csv':
        write_csv(folder/'observations.csv',arrays() if values is None else values)
    return folder


def test_npz_and_csv_round_trip_match_without_resampling(tmp_path):
    values=arrays()
    np.savez(tmp_path/'a.npz',**values)
    write_csv(tmp_path/'a.csv',values)
    a,b=load_standard_npz(tmp_path/'a.npz'),load_standard_csv(tmp_path/'a.csv')
    for name in values:
        assert np.array_equal(getattr(a,name),getattr(b,name))
        assert np.array_equal(getattr(a,name),values[name])


@pytest.mark.parametrize('fmt',['npz','csv'])
def test_dataset_report_is_small_json_summary(tmp_path,fmt):
    folder=write_dataset(tmp_path/fmt,metadata('observations.'+fmt))
    report=validate_dataset(folder)
    assert report['status']=='valid'
    assert report['shape']==[4,3]
    assert report['kind']=='synthetic'
    assert report['summary']['time_range']==[0.,.13]
    assert 'theta' not in report
    json.dumps(report,allow_nan=False)


def test_npz_extra_arrays_are_informational_and_not_unpickled(tmp_path):
    folder=write_dataset(tmp_path/'data')
    np.savez(folder/'observations.npz',**arrays(),notes=np.array([{'not':'loaded'}],dtype=object))
    report=validate_dataset(folder)
    assert report['extra_arrays']==['notes']


@pytest.mark.parametrize('replacement,match',[
    ({'schema_version':2},'schema_version'),({'schema_version':True},'schema_version'),
    ({'kind':'measured_truth'},'kind'),({'name':''},'name'),({'time_origin':' '},'time_origin'),
    ({'license':''},'license'),({'source':False},'source'),({'details':[]},'details'),
    ({'depth_positive':'up'},'depth_positive'),({'units':{'time':'hour','depth':'m','water_content':'fraction'}},'units'),
    ({'unexpected':1},'fields'),({'observations':'../elsewhere.npz'},'contained'),
    ({'observations':'not-found.npz'},'contained'),
])
def test_invalid_metadata_rejected(tmp_path,replacement,match):
    meta=metadata()
    meta.update(replacement)
    folder=write_dataset(tmp_path/'bad',meta)
    with pytest.raises(ValueError,match=match):validate_dataset(folder)


def test_metadata_missing_fields_and_invalid_yaml_rejected(tmp_path):
    folder=write_dataset(tmp_path/'bad')
    meta=metadata()
    del meta['source']
    (folder/'metadata.yaml').write_text(yaml.safe_dump(meta),encoding='utf-8')
    with pytest.raises(ValueError,match='fields'):validate_dataset(folder)
    (folder/'metadata.yaml').write_text('schema_version: [',encoding='utf-8')
    with pytest.raises(ValueError,match='YAML'):validate_dataset(folder)


@pytest.mark.parametrize('text,match',[
    ('timestamp,0.05m,0.1m\n0,.2,.3\n1,.3,.4','first column'),
    ('0.05m,time,0.1m\n.2,0,.3\n.3,1,.4','first column'),
    ('time,0.05m,extra\n0,.2,.3\n1,.3,.4','depth columns'),
    ('time,0.05m,0.05m\n0,.2,.3\n1,.3,.4','duplicate'),
    ('time,0.05m,0.050m\n0,.2,.3\n1,.3,.4','unique'),
    ('time,0.1m,0.05m\n0,.2,.3\n1,.3,.4','increasing'),
    ('time,-0.1m,0.05m\n0,.2,.3\n1,.3,.4','nonnegative'),
    ('time,nanm,0.05m\n0,.2,.3\n1,.3,.4','finite'),
    ('time,0.05m,0.1m\n0,.2,.3,1\n1,.3,.4','number of columns'),
    ('time,0.05m,0.1m\n0,,.3\n1,.3,.4','numeric'),
    ('time,0.05m,0.1m\n0,nan,.3\n1,.3,.4','non-finite'),
    ('time,0.05m,0.1m\n0,.2,.3\n0,.3,.4','increasing'),
    ('time,0.05m,0.1m\n1,.2,.3\n0,.3,.4','increasing'),
    ('time,0.05m,0.1m\n-1,.2,.3\n0,.3,.4','nonnegative'),
])
def test_bad_explicit_csv_rejected(tmp_path,text,match):
    path=tmp_path/'bad.csv'
    path.write_text(text,encoding='utf-8')
    with pytest.raises(ValueError,match=match):load_standard_csv(path)


@pytest.mark.parametrize('key,value,match',[
    ('times',np.array([0.,0.,.015,.13]),'increasing'),
    ('depths',np.array([-.05,.1,.25]),'nonnegative'),
    ('theta',np.array([[1.2]*3]*4),'fraction'),
    ('theta',np.array([[float('nan')]*3]*4),'finite'),
    ('theta',np.array([[True]*3]*4),'numeric'),
    ('times',np.array(['0','1','2','3']),'numeric'),
    ('times',np.array([3,2,1,0],dtype=np.uint64),'increasing'),
    ('depths',np.array([3,2,1],dtype=np.uint64),'increasing'),
])
def test_bad_npz_rejected(tmp_path,key,value,match):
    values=arrays()
    values[key]=value
    path=tmp_path/'bad.npz'
    np.savez(path,**values)
    with pytest.raises(ValueError,match=match):load_standard_npz(path)


def test_npz_requires_named_arrays(tmp_path):
    path=tmp_path/'bad.npz'
    np.savez(path,theta=arrays()['theta'])
    with pytest.raises(ValueError,match='required arrays'):load_standard_npz(path)


def test_explicit_csv_workflow_infers_depths_subsets_and_normalizes_percent(tmp_path):
    values=arrays()
    values['theta']=values['theta']*100
    path=tmp_path/'obs.csv'
    write_csv(path,values)
    cfg=load_experiment(dict(schema_version=2,data=dict(path=str(path),format='csv',
        minutes_per_row=200,units=dict(water_content='percent'))))
    obs=load_data(cfg)
    assert np.array_equal(obs.times,arrays()['times'])
    assert np.allclose(obs.theta,arrays()['theta'])
    cfg.data.depths=[.05,.25]
    subset=load_data(cfg)
    assert np.array_equal(subset.depths,np.array([.05,.25]))
    assert np.array_equal(subset.times,obs.times)
    assert np.allclose(subset.theta,obs.theta[:,[0,2]])


def test_implicit_csv_requires_depths_and_keeps_minutes_sampling(tmp_path):
    path=tmp_path/'sensor.csv'
    path.write_text('0.05m,0.1m\n.2,.3\n.25,.35\n.3,.4\n',encoding='utf-8')
    cfg=load_experiment(dict(schema_version=2,data=dict(path=str(path),format='csv')))
    with pytest.raises(ValueError,match='requires explicit data.depths'):load_data(cfg)
    cfg.data.depths=[.05,.1]
    cfg.data.minutes_per_row=5
    obs=load_data(cfg)
    assert np.array_equal(obs.times,np.array([0.,5.,10.])/1440)


def test_validate_data_cli_good_and_bad(tmp_path,capsys):
    folder=write_dataset(tmp_path/'good')
    main(['validate-data',str(folder)])
    assert json.loads(capsys.readouterr().out)['shape']==[4,3]
    (folder/'metadata.yaml').write_text('schema_version: 2',encoding='utf-8')
    with pytest.raises(SystemExit) as error:main(['validate-data',str(folder)])
    assert error.value.code!=0


def test_shared_validator_catches_array_mutation():
    obs=Observations(**arrays())
    obs.theta[0,0]=np.nan
    with pytest.raises(ValueError,match='finite'):validate_observations(obs)


def test_only_matching_sidecar_attaches_without_injecting_priors(tmp_path):
    meta=metadata()
    meta['details']={'known_coefficients':{'1':100},'prior':{'theta_z':1}}
    folder=write_dataset(tmp_path/'data',meta)
    cfg=load_experiment(dict(schema_version=2,data=dict(path=str(folder/'observations.npz'))))
    before=cfg.to_dict()
    report,record=_dataset_provenance(cfg)
    assert report['shape']==[4,3]
    assert record['path']==str((folder/'metadata.yaml').resolve())
    assert cfg.to_dict()==before
    write_csv(folder/'alternative.csv',arrays())
    cfg.data.path=str(folder/'alternative.csv')
    assert _dataset_provenance(cfg)==(None,None)


def test_matching_sidecar_conflicting_units_fail(tmp_path):
    folder=write_dataset(tmp_path/'data')
    cfg=load_experiment(dict(schema_version=2,data=dict(path=str(folder/'observations.npz'),units=dict(time='hour'))))
    with pytest.raises(ValueError,match='conflict'):_dataset_provenance(cfg)
