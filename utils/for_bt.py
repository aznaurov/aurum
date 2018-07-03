import pandas as pd
import numpy as np
import os
import bt
import cvxopt as opt
import sys
import os
from cvxopt import blas, solvers, sparse

import tensorflow as tf
import datetime 
from datetime import datetime, timedelta

from IPython.display import clear_output, Image, display, HTML

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))

def roundTime(dt, roundTo=5 * 60):
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds+roundTo/2) // roundTo * roundTo
    return dt + timedelta(0,rounding-seconds,-dt.microsecond)
    
class WeighTarget(bt.Algo):

    def __init__(self, target_weights):
        self.tw = target_weights

    def __call__(self, target):
        # get target weights on date target.now
        if target.now in self.tw.index:
            w = self.tw.loc[target.now]

            # save in temp - this will be used by the weighing algo
            # also dropping any na's just in case they pop up
            target.temp['weights'] = w.dropna()

        # return True because we want to keep on moving down the stack
        return True
    
def backtest_single(weight_series, price_series, name, my_comm, verbose=False):
    baseline = bt.Strategy(name, [WeighTarget(weight_series),
                                bt.algos.Rebalance()])

    result = bt.run(bt.Backtest(baseline, price_series, progress_bar=verbose, commissions=my_comm))
    return result

def calc_wghts_mpt_minstd(prices):
    returns = prices.pct_change()
    returns = returns.dropna()
    returns = returns.T
    n = len(returns)
    returns = np.asmatrix(returns)

    N = 20
    mus = [0.0001 * t/N  for t in range(N)]

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))

    q = opt.matrix(0.0, (n, 1))

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    A = opt.matrix([[A, pbar.T]])

    # Calculate efficient frontier weights using quadratic programming
    portfolios =  []
    for mu in mus:
        b = opt.matrix([[1.0, mu]])
        portfolio = solvers.qp(S, q, None, None, A, b)['x']
        portfolios.append(portfolio)

    ## CALCULATE RISKS AND RETURNS 
    means = [blas.dot(pbar, x) for x in portfolios]
    stds = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]

    #plt.plot(stds, means, 'go', linewidth=0.2)
    #plt.ylabel('mean')
    #plt.xlabel('std')
    #plt.show()
    #print(np.array(portfolios[np.argmin(stds)]).T[0])
    return np.array(portfolios[np.argmin(stds)]).T[0]