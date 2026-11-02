"""Fixed-support engine checks: correctness and pairing, not accuracy claims."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from deepgs.coefficients import build_coefficient_model, coefficient_from_spec
from deepgs.config import FieldConfig
from deepgs.contracts import SelectionResult
from deepgs.data import Observations
from deepgs.experiment import (BSplineConfig, InversionConfig, NetworkConfig,
    PolynomialConfig, TrainingConfig, CollocationConfig)
from deepgs.fixed_inversion import invert
from deepgs.inversion import InversionSettings, run_inversion, sample_collocation
from deepgs.networks import FieldNetwork, observation_bounds
from deepgs.pinn import train_pinn


def observations():
    t=np.linspace(0,1,20)
    z=np.array([.05,.15,.25])
    return Observations(t,z,.2+.05*np.sin(4*t)[:,None]*np.exp(-z)[None,:])


def settings(form='network'):
    cfg=InversionConfig(enabled=True,coefficient_form=form,seed=7,
        field=FieldConfig(hidden_layers=2,width=8,activation='silu',dtype='float64'),
        training=TrainingConfig(epochs=3,log_every=1),
        collocation=CollocationConfig(points=24))
    if form=='network':
        cfg.network=NetworkConfig(hidden_sizes=[5],theta_hidden=[5],depth_hidden=[3],fusion_hidden=[4])
    elif form=='polynomial':
        cfg.polynomial=PolynomialConfig(signs={'1':1,'theta_z':-1,'theta_zz':1,'theta_z^2':1,'theta_zzzz':1})
    else:
        cfg.bspline=BSplineConfig(n_internal_knots=2)
    return cfg


@pytest.mark.parametrize('form',['network','polynomial','bspline'])
def test_fixed_terms_matrix_shape_and_checkpoint_reconstruction(form):
    selection=SelectionResult.supplied(['1','theta_z','theta_zz'])
    result=invert(observations(),selection,settings(form))
    assert result.status=='completed'
    assert result.terms==selection.terms
    assert np.isfinite(result.metrics['theta_rmse'])
    assert len(result.history)==3
    assert all(np.isfinite(v) for row in result.history for v in row.values())
    low,high=result.domain['theta_bounds']
    theta=torch.linspace(low,high,17,dtype=torch.float64)[:,None]
    reconstructed=coefficient_from_spec(result.coefficients.spec).double()
    reconstructed.load_state_dict(result.coefficients.state_dict())
    assert result.coefficients(theta).shape==(17,3)
    assert torch.equal(result.coefficients(theta),reconstructed(theta))


def test_forms_have_identical_field_initialization_and_collocation():
    results=[invert(observations(),SelectionResult.supplied(['theta_z','theta_zz']),settings(form)) for form in ('network','polynomial','bspline')]
    assert len({r.metadata['initial_field_sha256'] for r in results})==1
    assert len({r.metadata['collocation_sha256'] for r in results})==1


def test_engine_leaves_caller_torch_random_state_unchanged():
    torch.manual_seed(83)
    before=torch.random.get_rng_state().clone()
    invert(observations(),SelectionResult.supplied(['theta_zz']),settings())
    assert torch.equal(before,torch.random.get_rng_state())


@pytest.mark.parametrize('status,expected',[('empty','skipped_empty'),('no_knee','skipped_no_structure')])
def test_invalid_or_empty_structure_skips_training(status,expected):
    selection=SelectionResult(status,(),(),None,(),'test')
    result=invert(observations(),selection,settings())
    assert result.status==expected
    assert result.field is None


def test_missing_prior_is_reported_without_deleting_term():
    cfg=settings('polynomial')
    cfg.polynomial.signs={'theta_z':-1}
    selection=SelectionResult.supplied(['theta_z','theta_zz'])
    result=invert(observations(),selection,cfg)
    assert result.status=='blocked_missing_prior'
    assert result.terms==selection.terms
    assert result.metadata['missing_signs']==['theta_zz']
    assert result.field is None


@pytest.mark.parametrize('architecture',['additive','fusion'])
def test_heterogeneous_network_uses_depth(architecture):
    cfg=settings()
    cfg.network.architecture=architecture
    result=invert(observations(),SelectionResult.supplied(['1','theta_zz']),cfg,soil='heterogeneous')
    theta=torch.full((10,1),.2,dtype=torch.float64)
    z=torch.linspace(.05,.25,10,dtype=torch.float64)[:,None]
    values=result.coefficients(theta,z)
    assert values.shape==(10,2)
    assert not torch.allclose(values[0],values[-1])
    with pytest.raises(ValueError,match='require'):
        result.coefficients(theta)


@pytest.mark.parametrize('form',['network','polynomial','bspline'])
def test_constant_only_and_fourth_order_support_run(form):
    result=invert(observations(),SelectionResult.supplied(['1']),settings(form))
    assert result.status=='completed'
    result=invert(observations(),SelectionResult.supplied(['theta_zzzz']),settings(form))
    assert result.status=='completed'


def test_bad_domain_and_unknown_terms_are_rejected():
    cfg=settings()
    cfg.domain.theta_bounds=[.2,.21]
    with pytest.raises(ValueError,match='contain'):
        invert(observations(),SelectionResult.supplied(['theta_z']),cfg)
    with pytest.raises(ValueError,match='term IDs'):
        invert(observations(),SelectionResult.supplied(['madeup']),settings())


def test_new_polynomial_replays_legacy_with_explicitly_matched_settings():
    obs=observations()
    cfg=settings('polynomial')
    terms=['theta_z','theta_zz','theta_z^2']
    new=invert(obs,SelectionResult.supplied(terms),cfg)
    oldcfg=InversionSettings(coefficient_form='polynomial',hidden_sizes=(8,8),epochs=3,
        collocation_points=24,log_every=1)
    old=run_inversion(obs,terms,[-1,1,1],torch.device('cpu'),oldcfg,seed=7)
    for key,value in new.field.state_dict().items():
        assert torch.allclose(value,old['field'].state_dict()[key],atol=1e-12,rtol=1e-12)
    assert np.allclose(new.coefficients.controls.numpy(),old['controls'],atol=1e-12,rtol=1e-12)


def test_network_replays_legacy_training_kernel_with_matched_initialization():
    obs=observations()
    cfg=settings()
    terms=['theta_z','theta_zz']
    new=invert(obs,SelectionResult.supplied(terms),cfg)
    bounds=tuple(new.domain['theta_bounds'])
    with torch.random.fork_rng():
        torch.manual_seed(cfg.seed)
        field=FieldNetwork([8,8],'silu',observation_bounds(obs),bounds,xavier=True).double()
        torch.manual_seed(cfg.seed+1000003)
        coefficients=build_coefficient_model(terms,'homogeneous',cfg,bounds).double()
    collocation=sample_collocation(obs,InversionSettings(collocation_points=24),cfg.seed+97)
    optimizer=torch.optim.Adam(list(field.parameters())+list(coefficients.parameters()),lr=.001)
    train_pinn(field,coefficients.branches,terms,obs,collocation,3,100,1,optimizer,log_every=1)
    for key,value in new.field.state_dict().items():
        assert torch.allclose(value,field.state_dict()[key],atol=1e-12,rtol=1e-12)
    for key,value in new.coefficients.state_dict().items():
        assert torch.allclose(value,coefficients.state_dict()[key],atol=1e-12,rtol=1e-12)
