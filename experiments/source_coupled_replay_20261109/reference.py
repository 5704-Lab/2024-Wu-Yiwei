"""CPU reference operators for source lifting, paired replay and cycle gating.

No image model is loaded. Continuous-error constants are supplied assumptions;
finite source residuals are measured directly. Semantic success is never inferred
from source reconstruction or from the candidate status.
"""
from dataclasses import dataclass
import math
from typing import Callable
import numpy as np


class BudgetExhausted(RuntimeError):
    pass


class CountedField:
    def __init__(self, function: Callable, budget=None):
        self.function=function
        self.budget=budget
        self.calls=0

    def __call__(self, state, time):
        if self.budget is not None and self.calls>=self.budget:
            raise BudgetExhausted('field call budget exhausted')
        self.calls+=1
        value=np.asarray(self.function(state,time),dtype=np.float64)
        if value.shape!=state.shape or not np.all(np.isfinite(value)):
            raise ValueError('field output shape/finite check failed')
        return value


def rms(value):
    return float(np.sqrt(np.mean(np.asarray(value,dtype=np.float64)**2)))


def state_array(value):
    value=np.asarray(value,dtype=np.float64)
    if value.size==0 or not np.all(np.isfinite(value)):
        raise ValueError('nonempty finite state required')
    return value


def step(field,state,time,increment,method='euler'):
    k1=field(state,time)
    if method=='euler':
        return state+increment*k1
    if method=='heun':
        k2=field(state+increment*k1,time+increment)
        return state+increment*(k1+k2)/2
    raise ValueError('method must be euler or heun')


def integrate(field,initial,times,method='euler'):
    state=state_array(initial).copy()
    times=np.asarray(times,dtype=float)
    if times.ndim!=1 or len(times)<2 or not np.all(np.isfinite(times)):
        raise ValueError('finite time vector required')
    diffs=np.diff(times)
    if not (np.all(diffs>0) or np.all(diffs<0)):
        raise ValueError('time grid must be strictly monotone')
    history=[state.copy()]
    for time,increment in zip(times[:-1],diffs):
        state=state_array(step(field,state,time,increment,method))
        history.append(state.copy())
    return np.stack(history)


def local_lift(source,horizon,field,order=1):
    source=state_array(source)
    if not np.isfinite(horizon) or not 0<horizon<1:
        raise ValueError('horizon must lie strictly between zero and one')
    if order==1:
        return source+horizon*field(source,0.)
    if order==2:
        return step(field,source,0.,horizon,'heun')
    raise ValueError('order must be one or two')


@dataclass
class DenseHeunSegment:
    time: float
    length: float
    source: np.ndarray
    k1: np.ndarray
    k2: np.ndarray

    def evaluate(self,offset):
        if not 0<=offset<=self.length:
            raise ValueError('offset outside segment')
        q=self.source+offset*self.k1+offset**2/(2*self.length)*(self.k2-self.k1)
        derivative=self.k1+offset/self.length*(self.k2-self.k1)
        return q,derivative


def dense_heun(source,time,length,field):
    source=state_array(source)
    if not np.isfinite(length) or length<=0:
        raise ValueError('positive segment length required')
    k1=field(source,time)
    k2=field(source+length*k1,time+length)
    return DenseHeunSegment(time,length,source,k1,k2)


def local_linear_error_bound(time,state_L,time_L,velocity_norm):
    if min(time,state_L,time_L,velocity_norm)<0 or not np.all(np.isfinite(
            [time,state_L,time_L,velocity_norm])):
        raise ValueError('finite nonnegative inputs required')
    K=state_L*velocity_norm+time_L
    if state_L==0:
        return K*time*time/2
    z=state_L*time
    remainder=(sum(z**k/math.factorial(k) for k in range(2,11))
               if z<1e-3 else math.expm1(z)-z)
    return K*remainder/state_L**2


def piecewise_defect_bound(lengths,residual_sup_bounds,state_L,initial_error=0.):
    lengths=np.asarray(lengths,dtype=float)
    residuals=np.asarray(residual_sup_bounds,dtype=float)
    if lengths.ndim!=1 or lengths.shape!=residuals.shape or np.any(lengths<=0):
        raise ValueError('matching positive segment lengths required')
    if min(state_L,initial_error)<0 or np.any(residuals<0) or not np.all(np.isfinite(
            np.r_[lengths,residuals,state_L,initial_error])):
        raise ValueError('finite nonnegative error bounds required')
    error=initial_error
    history=[error]
    for length,delta in zip(lengths,residuals):
        phi=math.expm1(state_L*length)/state_L if state_L else length
        error=math.exp(state_L*length)*error+delta*phi
        history.append(error)
    return np.asarray(history)


@dataclass
class ReplayResult:
    status: str
    candidate: np.ndarray | None
    source_cycle_rms: float | None
    source_history: np.ndarray | None
    target_history: np.ndarray | None
    clean_history: np.ndarray | None
    semantic_success: bool=False


def paired_replay(source,lifted,source_field,target_field,times,
                  method='euler',cycle_rms_max=.02):
    source,lifted=state_array(source),state_array(lifted)
    times=np.asarray(times,dtype=float)
    if source.shape!=lifted.shape or not np.isfinite(cycle_rms_max) or cycle_rms_max<0:
        raise ValueError('matching states and a finite cycle threshold required')
    if times.ndim!=1 or len(times)<2 or times[-1]!=0 or not np.all(np.diff(times)<0):
        raise ValueError('replay grid must decrease to zero')
    try:
        hs=integrate(source_field,lifted,times,method)
        cycle=rms(hs[-1]-source)
        if cycle>cycle_rms_max:
            return ReplayResult('rejected_source_cycle',None,cycle,hs,None,None)
        ht=integrate(target_field,lifted,times,method)
        # Keep this grouping: identical branch states subtract to exact zero.
        clean=source[None,...]+(ht-hs)
        return ReplayResult('candidate_requires_task_evaluation',clean[-1],cycle,hs,ht,clean)
    except BudgetExhausted:
        return ReplayResult('budget_exhausted',None,None,None,None,None)


def cycle_to_edit_bound(cycle_norm,difference_map_L,inverse_source_L):
    values=[cycle_norm,difference_map_L,inverse_source_L]
    if min(values)<0 or not np.all(np.isfinite(values)):
        raise ValueError('finite nonnegative conditional constants required')
    return cycle_norm*difference_map_L*inverse_source_L


def invert_euler_replay(source,field,times,tolerance=1e-10,max_iterations=100):
    """High-cost reference: invert each discrete backward Euler map by fixed point.

Convergence requires h*L<1. Residuals are checked, but this function does not
automatically certify a neural-network Lipschitz bound or inversion uniqueness.
"""
    times=np.asarray(times,dtype=float)
    if times.ndim!=1 or len(times)<2 or times[-1]!=0 or not np.all(np.diff(times)<0):
        raise ValueError('decreasing generation grid ending at zero required')
    if tolerance<=0 or max_iterations<1:
        raise ValueError('positive tolerance and iteration budget required')
    cleaner=state_array(source).copy()
    records=[]
    for j in range(len(times)-2,-1,-1):
        time=times[j];length=times[j]-times[j+1]
        q=cleaner.copy()
        converged=False
        for iteration in range(max_iterations):
            velocity=field(q,time)
            residual=q-length*velocity-cleaner
            error=rms(residual)
            if error<=tolerance:
                converged=True
                break
            q=cleaner+length*velocity
        records.append(dict(index=j,iterations=iteration+1,residual_rms=error,converged=converged))
        if not converged:
            return None,records
        cleaner=q
    return cleaner,records
