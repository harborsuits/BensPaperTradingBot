import numpy as np

def benjamini_hochberg(pvals, alpha=0.10):
    pvals = np.asarray(pvals)
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = alpha * (np.arange(1, m+1)/m)
    passed = ranked <= thresh
    k = np.where(passed)[0].max()+1 if passed.any() else 0
    cutoff = ranked[k-1] if k>0 else 0.0
    qvals = np.empty_like(ranked)
    prev = 1.0
    for i in range(m-1, -1, -1):
        qvals[i] = min(prev, ranked[i]*m/(i+1))
        prev = qvals[i]
    out = np.empty_like(qvals)
    out[order] = qvals
    return cutoff, out  # use q<=0.10 as pass

def bootstrap_sharpe_ci(returns, n=1000, alpha=0.05):
    rs = np.random.default_rng(42)
    boots = []
    r = np.asarray(returns)
    r = r[~np.isnan(r)]
    for _ in range(n):
        s = rs.choice(r, size=len(r), replace=True)
        boots.append(np.mean(s)/np.std(s, ddof=1) if np.std(s, ddof=1)>0 else 0.0)
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return lo, hi
