import numpy as np
class GaussianHMM:
    """Supervised HMM with diagonal-Gaussian emissions, implemented in numpy."""
    def __init__(self,n_states,var_floor=1e-2,alpha=1.0):
        self.K=n_states; self.vf=var_floor; self.alpha=alpha
    def fit_emissions(self,X,z):
        D=X.shape[1]; self.mu=np.zeros((self.K,D)); self.var=np.zeros((self.K,D))
        for k in range(self.K):
            Xk=X[z==k]; self.mu[k]=Xk.mean(0); self.var[k]=Xk.var(0)+self.vf
        return self
    def fit_transitions(self,z_seqs):
        C=np.full((self.K,self.K),self.alpha); P=np.full(self.K,self.alpha)
        for z in z_seqs:
            P[z[0]]+=1
            for a,b in zip(z[:-1],z[1:]): C[a,b]+=1
        self.A=C/C.sum(1,keepdims=True); self.pi=P/P.sum(); return self
    def log_emission(self,X):
        d=X[:,None,:]-self.mu[None]
        return -0.5*(np.log(2*np.pi*self.var)[None]+d**2/self.var[None]).sum(2)   # (T,K)
    def viterbi(self,X):
        T=len(X); lb=self.log_emission(X); lA=np.log(self.A); 
        dp=np.zeros((T,self.K)); bp=np.zeros((T,self.K),int); dp[0]=np.log(self.pi)+lb[0]
        for t in range(1,T):
            s=dp[t-1][:,None]+lA; bp[t]=s.argmax(0); dp[t]=s.max(0)+lb[t]
        path=np.zeros(T,int); path[-1]=dp[-1].argmax()
        for t in range(T-2,-1,-1): path[t]=bp[t+1,path[t+1]]
        return path,dp[-1].max()
    def predict_independent(self,X):   # no transitions: baseline classifier
        return self.log_emission(X).argmax(1)
