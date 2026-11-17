"""Weighted kernel ridge classification with a true unpenalized intercept."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor,cho_solve,eigh
from scipy.special import softmax
from .multiview_bank import array_hash


class KernelData:
    """Train-only statistics; query rows never determine scales or centering."""
    def __init__(self,train,labels,query,n_classes,bandwidth_multiplier=1.,floor=1e-12):
        x=np.asarray(train,dtype=np.float64);q=np.asarray(query,dtype=np.float64);y=np.asarray(labels)
        if (x.ndim!=2 or q.ndim!=2 or x.shape[1]!=q.shape[1] or y.shape!=(len(x),)
                or not np.array_equal(np.unique(y),np.arange(n_classes)) or not np.isfinite(x).all()
                or not np.isfinite(q).all() or bandwidth_multiplier<=0 or floor<=0):raise ValueError('Invalid kernel membership/data')
        self.weights=len(y)/(n_classes*np.bincount(y,minlength=n_classes)[y]);self.r=self.weights/self.weights.sum()
        self.y=2*np.eye(n_classes)[y]-1
        self.y_mean=self.r@self.y;self.y_centered=self.y-self.y_mean
        # Translation improves stability; it has no effect on Gaussian distances.
        self.feature_mean=self.r@x;x=x-self.feature_mean;q=q-self.feature_mean
        self.linear=x@x.T;self.cross=q@x.T
        diag=np.diag(self.linear);self.distances=np.maximum(diag[:,None]+diag[None,:]-2*self.linear,0)
        np.fill_diagonal(self.distances,0)
        self.query_distances=np.maximum(np.sum(q*q,axis=1)[:,None]+diag[None,:]-2*self.cross,0)
        positive=self.distances[np.triu_indices(len(x),1)];positive=positive[positive>floor]
        self.bandwidth_squared=float(np.median(positive))*bandwidth_multiplier if len(positive) else 1.
        self.energy=float(self.r@diag)
        if self.energy<=floor:self.energy=1.
        self.state=dict(training_features_sha256=array_hash(np.asarray(train,dtype=np.float64)),training_labels_sha256=array_hash(y),
                        feature_mean_sha256=array_hash(self.feature_mean),weights_sha256=array_hash(self.weights),
                        bandwidth_squared=self.bandwidth_squared,linear_energy=self.energy,n_train=len(x),n_features=x.shape[1],n_classes=n_classes)
        self.cache={}

    def kernel(self,weight):
        if not 0<=weight<=1:raise ValueError('Invalid kernel mixture')
        if weight not in self.cache:
            k=(1-weight)*self.linear;cross=(1-weight)*self.cross
            if weight:
                k=k+weight*self.energy*np.exp(-self.distances/(2*self.bandwidth_squared))
                cross=cross+weight*self.energy*np.exp(-self.query_distances/(2*self.bandwidth_squared))
            smallest=float(eigh(k,subset_by_index=[0,0],eigvals_only=True,check_finite=True)[0])
            if smallest < -1e-8*max(1.,float(np.max(np.diag(k)))):raise ValueError('Kernel is not positive semidefinite')
            row=k@self.r;grand=float(self.r@row)
            kc=k-row[:,None]-row[None,:]+grand
            qc=cross-(cross@self.r)[:,None]-row[None,:]+grand
            state=dict(self.state,nonlinear_weight=weight,raw_kernel_sha256=array_hash(k),centered_kernel_sha256=array_hash(kc),
                       kernel_row_mean_sha256=array_hash(row),kernel_grand_mean=grand)
            self.cache[weight]=(kc,qc,state)
        return self.cache[weight]


@dataclass
class KernelModel:
    alpha: float
    weight: float
    state: dict
    dual: np.ndarray
    intercept: np.ndarray

    @classmethod
    def fit(cls,data,alpha,weight):
        if not np.isfinite(alpha) or alpha<=0:raise ValueError('Invalid ridge alpha')
        k,_,state=data.kernel(weight);sqrt=np.sqrt(data.weights)
        a=k*sqrt[:,None]*sqrt[None,:];a=a.copy();a.flat[::len(a)+1]+=alpha
        dual=sqrt[:,None]*cho_solve(cho_factor(a,lower=True),sqrt[:,None]*data.y_centered)
        return cls(alpha,weight,state,dual,data.y_mean.copy())

    def scores(self,data):
        _,cross,state=data.kernel(self.weight)
        if state!=self.state:raise ValueError('Kernel fitted state changed')
        scores=cross@self.dual+self.intercept
        if not np.isfinite(scores).all():raise ValueError('Nonfinite kernel logits')
        return scores

    def predict(self,data):return softmax(self.scores(data),axis=1)
