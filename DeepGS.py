# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np
import time
import pandas as pd
import chaospy
from numpy.linalg import norm as Norm


class van_G_para:
    def __init__(self,texture):

        self.Ks = 0.2496
        self.theta_s= 0.43
        self.theta_r= 0.078
        self.alpha= 3.6
        self.n= 1.56
            
    def Se(self,theta):
        Se = (theta - self.theta_r)/ (self.theta_s-self.theta_r)
        return Se
    
    def VG_dz (self,theta):
        Se =  self.Se(theta)
        m = 1 -1/self.n
        a = 1- np.power(Se,1/m)
        coef = - 0.5 * self.Ks * np.power(Se,-0.5) * np.square(1 - np.power(a,m))- 2 * self.Ks * np.power(Se,-0.5+1/m) * (1-np.power(a,m)) * np.power(a,m-1)
        coef = (1/(self.theta_s - self.theta_r)) * coef
        return coef
    
    def VG_d2z (self,theta):
        Se =  self.Se(theta)
        m = 1 -1/self.n
        a = 1- np.power(Se,1/m)

        coef = np.power(Se,0.5-1/m-1) * np.square(1-np.power(a,m)) * np.power(np.power(Se,-1/m),1/self.n-1)
        coef = (self.Ks/((self.theta_s - self.theta_r)*(self.alpha * self.n *m))) * coef
        return coef
    
    def VG_dz2 (self,theta):
        Se =  self.Se(theta)
        m = 1 -1/self.n
        a = 1-np.power(Se,1/m)
        D = self.VG_d2z(theta)
        coef = (0.5-1/m-1)*np.power(Se,-1)+2*np.power(Se,1/m-1)*np.power(1-np.power(a,m),-1)*np.power(a,m-1)+(1/m)*(1-1/self.n)*np.power(Se,-1/m-1)*np.power(np.power(Se,-1/m)-1,-1)
        coef  = (D/(self.theta_s - self.theta_r))* coef
        return coef
    

def data_group(x,y,u,win_size = 0.001, min_lim = 200):

    n,d = x.shape
    
    norm_x = np.ones((d,1))
    for j in range(0,d):
        norm_x[j] = (np.linalg.norm(x[:,j],2))
        x[:,j] = x[:,j]/norm_x[j]
    
    norm_y = np.linalg.norm(y,2)
    y = y/norm_y
    
    idx = np.argsort(np.squeeze(u))
    u = u[idx]
    x = x[idx]
    y = y[idx]
        
    upper = np.max(u)
    lower = np.min(u)
    idx=[]
        
    _lower = lower
    _upper = lower + win_size
    
    iter = 0
    while True:
        
        _idx = np.where((u>=_lower) & (u<=_upper))[0]
        if _idx[-1] + min_lim < np.where(u==upper)[0][0]:
            if  _idx.shape[0] > min_lim:
                idx.append(_idx)
                
            else:
                _idx = np.arange(_idx[0],_idx[0]+min_lim)
                idx.append(_idx)
                
            iter +=1
        else:
            _idx = np.arange(_idx[0],np.where(u==upper)[0][0])
            idx.append(_idx)
            
            break
        
        _lower = u[_idx[-1],0]
        _upper = _lower + win_size
    
    
    lower_control = int(np.ceil(0.01*len(idx)))
    upper_control = int(np.floor(0.99*len(idx)))
    
    
    lower_idx = np.hstack(idx[:lower_control])
    upper_idx = np.hstack(idx[upper_control:])

    idx = [lower_idx] + idx[lower_control:upper_control] + [upper_idx]
        
    Theta_grouped=[]
    Ut_grouped = []
    Uz_grouped = []
        
    for i , _idx in enumerate(idx):
            _x = x[_idx,:]
            _y = y[_idx]
            
            Theta_grouped.append(np.mean(u[_idx]))
            Uz_grouped.append(_x)
            Ut_grouped.append(_y)
    
    return Theta_grouped,Uz_grouped, Ut_grouped,norm_x,norm_y


def Ridge(A,b,lam):
    if lam != 0: return np.linalg.solve(A.T.dot(A)+lam*np.eye(A.shape[1]), A.T.dot(b))
    else: return np.linalg.lstsq(A, b ,rcond=None)[0]
    

def GS(Xs, ys, tol, lam, maxit = 1, verbose = True):

    if len(Xs) != len(ys): raise Exception('Number of Xs and ys mismatch')
    if len(set([X.shape[1] for X in Xs])) != 1: 
        raise Exception('Number of coefficients inconsistent across timesteps')
        
    d = Xs[0].shape[1]
    m = len(Xs)
    
    W = np.hstack([Ridge(X,y,lam) for [X,y] in zip(Xs,ys)])
    
    num_relevant = d
    biginds = [i for i in range(d) if np.linalg.norm(W[i,:]) > tol]
    
    for j in range(maxit):
        
        smallinds = [i for i in range(d) if np.linalg.norm(W[i,:]) < tol]
        new_biginds = [i for i in range(d) if i not in smallinds]
            
        if num_relevant == len(new_biginds): j = maxit-1
        else: num_relevant = len(new_biginds)
            
        if len(new_biginds) == 0:
            if j == 0 and verbose: 
                print("Tolerance too high - all coefficients set below tolerance")
            break
        biginds = new_biginds
        
        for i in smallinds:
            W[i,:] = np.zeros(m)
        if j != maxit -1:
            for i in range(m):
                W[biginds,i] = Ridge(Xs[i][:, biginds], ys[i], lam).reshape(len(biginds))
        else: 
            for i in range(m):
                W[biginds,i] = np.linalg.lstsq(Xs[i][:, biginds],ys[i],rcond=None)[0].reshape(len(biginds))
                
    return W


def TrainGS(As, bs,  norm_x, norm_y,num_tols=1000, lam = 10**-2):

    np.random.seed(0)

    n,D = As[0].shape
    
    x_ridge = np.hstack([Ridge(A,b,lam) for (A,b) in zip(As, bs)])
    max_tol = np.max([Norm(x_ridge[j,:]) for j in range(x_ridge.shape[0])])
    min_tol = np.min([Norm(x_ridge[j,:]) for j in range(x_ridge.shape[0])])
    Tol = [0]+ [np.exp(alpha) for alpha in np.linspace(np.log(min_tol), np.log(max_tol), num_tols)][:-1]

    X = []
    Losses = []

    for tol in Tol:
        x = GS(As,bs,tol,lam)
        X.append(x)
        Losses.append(AIC(As,bs,x))
        
    for x in X:
        for i in range(D):
                x[i,:] = x[i,:]/norm_x[i]*norm_y
                
                
    return X,Tol,Losses


def AIC(As,bs,x,epsilon= 10**-1.5):

    D,m = x.shape
    n,_ = As[0].shape
    N = n*m
    rss = np.sum([np.linalg.norm(bs[j] - As[j].dot(x[:,j].reshape(D,1)))**2 for j in range(m)])
    k = np.count_nonzero(x)/m
    
    return N * np.log(rss/N + epsilon ) + 2 * k

class DeepGS:
    def __init__(self, train_points, res_points, theta, layers_theta, layers_coef, sparse):
        self.z = train_points[:,0:1]
        self.t = train_points[:,1:2]
        self.theta = theta
        
        self.res_z = res_points[:,0:1]
        self.res_t = res_points[:,1:2]
        self.res_number = self.res_z.shape[0]
        
        self.layers_coef = layers_coef
        self.layers_theta = layers_theta
        
        self.sparse_tf = tf.placeholder(tf.float32, shape=self.sparse.shape)
        
        self.weights_theta, self.biases_theta = self.initialize_NN(layers_theta)
        
        for i in range (10):
            exec ('self.weights_coef%s, self.biases_coef%s = self.initialize_NN(layers_coef)'%(i,i))


        config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth=True
        
        self.sess = tf.Session(config=config)

        self.z_tf = tf.placeholder(tf.float32, shape = [None, 1])
        self.t_tf = tf.placeholder(tf.float32, shape = [None, 1])
        
        self.res_z_tf = tf.placeholder(tf.float32, shape = [None, 1])
        self.res_t_tf = tf.placeholder(tf.float32, shape = [None, 1])
        self.theta_tf = tf.placeholder(tf.float32, shape = [None, 1])
        
        self.constant_val_data = np.array(1.0)
        self.constant_tf_data = tf.placeholder(tf.float32, shape=self.constant_val_data.shape)
        
        self.constant_val_res = np.array(0)
        self.constant_tf_res = tf.placeholder(tf.float32, shape=self.constant_val_res.shape)
        
        self.data_loss =  self.constant_tf_data * tf.reduce_mean(tf.square(self.net_theta(self.z_tf, self.t_tf) - self.theta_tf))
        
        self.res = self.residual_loss(self.res_z_tf, self.res_t_tf)
        self.res_loss = self.constant_tf_res * tf.reduce_mean(self.res)
        
        self.loss =  self.data_loss + self.res_loss

        self.z_pred_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t_pred_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.candidate_pred = self.net_candidate(self.z_pred_tf, self.t_pred_tf)
        self.theta_coef_pred = tf.placeholder(tf.float32, shape=[None, 1])
        
        for i in range (10):
            exec ('self.coef%s = self.net_coef(self.theta_coef_pred, self.weights_coef%s, self.biases_coef%s)'%(i,i,i))

        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(1e-3, self.global_step, 1000, 0.95, staircase=False)
        self.LBFGSB_op = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                        method = 'L-BFGS-B',
                                                        options = {'maxiter': 500,
                                                                    'maxfun': 50000,
                                                                    'maxcor': 50,
                                                                    'maxls': 50,
                                                                    'ftol' : 1.0 * np.finfo(float).eps})
        
        self.Adam_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,global_step=self.global_step)

        self.loss_data_log = []
        self.loss_res_log = []
        self.loss_log = []
        self.iter_log = []
        

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev = xavier_stddev), dtype = tf.float32)
    
    def initialize_NN(self, layers):
        weights = []
        biases = []
        
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            
            weights.append(W)
            biases.append(b)
        return weights, biases
    
    
    def net_theta(self, z, t):
        z = 2.0*(z - np.min(self.z))/(np.max(self.z) - np.min(self.z)) - 1.0
        t = 2.0*(t - np.min(self.t))/(np.max(self.t) - np.min(self.t)) - 1.0
        
        X = tf.concat([z, t],1)
        weights = self.weights_theta
        biases = self.biases_theta
        num_layers = len(weights) + 1
        
        H = X

        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))

        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        
        return Y

    def net_coef(self, theta, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(theta - np.min(self.theta))/(np.max(self.theta) - np.min(self.theta)) - 1.0
        
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
                
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        
        return Y
    
    def net_candidate(self, z, t):
        u = self.net_theta(z, t)
        ut = tf.gradients(u, t)[0]
        ux = tf.gradients(u, z)[0]
        u2x = tf.gradients(ux, z)[0]
        u3x = tf.gradients(u2x, z)[0]
        ux_2 = tf.square(ux)
        u2x_2 = tf.square(u2x)
        u3x_2 = tf.square(u3x)
        ux_u2x = ux * u2x
        ux_u3x = ux * u3x
        u2x_u3x = u2x * u3x
        constant = tf.ones_like(u)
        candidate = [u, constant, ux, u2x, u3x, ux_2, u2x_2, u3x_2, ux_u2x, ux_u3x, u2x_u3x, ut]
        return candidate
    
    
    def residual_loss(self, z, t):
        candidate = self.net_candidate(z, t)
        res_theta = self.net_theta(z, t)
        spatial_derivative = tf.concat(candidate[1:-1],1)
        coef0 = self.net_coef(res_theta,self.weights_coef0, self.biases_coef0)
        coef1 = self.net_coef(res_theta,self.weights_coef1, self.biases_coef1)
        coef2 = self.net_coef(res_theta,self.weights_coef2, self.biases_coef2)
        coef3 = self.net_coef(res_theta,self.weights_coef3, self.biases_coef3)
        coef4 = self.net_coef(res_theta,self.weights_coef4, self.biases_coef4)
        coef5 = self.net_coef(res_theta,self.weights_coef5, self.biases_coef5)
        coef6 = self.net_coef(res_theta,self.weights_coef6, self.biases_coef6)
        coef7 = self.net_coef(res_theta,self.weights_coef7, self.biases_coef7)
        coef8 = self.net_coef(res_theta,self.weights_coef8, self.biases_coef8)
        coef9 = self.net_coef(res_theta,self.weights_coef9, self.biases_coef9)
        coef = tf.concat([coef0,coef1,coef2,coef3,coef4,coef5,coef6,coef7,coef8,coef9],1)
        
        residual_loss = tf.square(candidate[-1]-tf.matmul(tf.math.multiply(coef,spatial_derivative),self.sparse_tf))
            

        return residual_loss
    
    def train(self, max_iter, sparse):
        start_time = time.time()
        Total_Time = 0
        
        for it in range(max_iter):
            tf_dict = {self.z_tf: self.z,
                       self.t_tf: self.t,
                       self.theta_tf: self.theta,
                       self.res_t_tf: self.res_t,
                       self.res_z_tf: self.res_z,
                       self.constant_tf_data: self.constant_val_data,
                       self.constant_tf_res :  self.constant_val_res,
                       self.sparse_tf: sparse}
            
            self.sess.run(self.Adam_op, tf_dict)
            
            if it % 100 == 0:

                    loss_value = self.sess.run(self.loss, tf_dict)
                    
                    data_loss_value = self.sess.run(self.data_loss, {self.z_tf: self.z, self.t_tf: self.t, self.theta_tf: self.theta,self.constant_tf_data: self.constant_val_data})
                    res_loss_value = self.sess.run(self.res_loss, {self.res_t_tf: self.res_t, self.res_z_tf: self.res_z, self.sparse_tf: self.sparse,self.constant_tf_res :  self.constant_val_res})
                    
                 
                    elapsed = time.time() - start_time
                    
                    Total_Time += elapsed
                    print('Iter: %d, Loss: %.2e, L_data: %.2e, L_res: %.2e, Time: %.2f, Total: %.1f Minutes' %
                          ( it,  loss_value, data_loss_value, res_loss_value, elapsed, Total_Time/60))

                    start_time = time.time()
                    
                    self.loss_data_log.append(data_loss_value)
                    self.loss_res_log.append(res_loss_value)
                    self.loss_log.append(loss_value)
                    self.iter_log.append(it)
                    

    def pre_train(self, max_iter, sparse):
        start_time = time.time()
        Total_Time = 0
        
        for it in range(max_iter):
            tf_dict = {self.z_tf: self.z,
                       self.t_tf: self.t,
                       self.theta_tf: self.theta,
                       self.res_t_tf: self.res_t,
                       self.res_z_tf: self.res_z,
                       self.constant_tf_data: np.array(1.0),
                       self.constant_tf_res : np.array(0.0),
                       self.sparse_tf: sparse}
            
            self.sess.run(self.Adam_op, tf_dict)
            
            if it % 100 == 0:

                    loss_value = self.sess.run(self.loss, tf_dict)
                    
                    data_loss_value = self.sess.run(self.data_loss, {self.z_tf: self.z, self.t_tf: self.t, self.theta_tf: self.theta,self.constant_tf_data: self.constant_val_data})
                    res_loss_value = self.sess.run(self.res_loss, {self.res_t_tf: self.res_t, self.res_z_tf: self.res_z, self.sparse_tf: self.sparse,self.constant_tf_res :  self.constant_val_res})
                    
                 
                    elapsed = time.time() - start_time
                    
                    Total_Time += elapsed
                    print('Iter: %d, Loss: %.2e, L_data: %.2e, L_res: %.2e, Time: %.2f, Total: %.1f Minutes' %
                          ( it,  loss_value, data_loss_value, res_loss_value, elapsed, Total_Time/60))

                    start_time = time.time()
                    
                    self.loss_data_log.append(data_loss_value)
                    self.loss_res_log.append(res_loss_value)
                    self.loss_log.append(loss_value)
                    self.iter_log.append(it)
                

                    
    def callback(self, loss):
        print('Loss: ', loss)

    def predict(self, X_star):
        f_star = self.sess.run(self.candidate_pred, {self.z_pred_tf: X_star[:,0:1], self.t_pred_tf: X_star[:,1:2]})
        
        return f_star
    


def mesh_2d(depth,day):
    X,T = np.meshgrid(depth,day)
    points = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    
    return points

def cvt_respoints(distribution,start_x,end_x,start_t,end_t):
    res_points = distribution.T
    res_points[:,0] *= end_x - start_x
    res_points[:,0] += start_x
    
    res_points[:,1] *= end_t - start_t
    res_points[:,1] += start_t
    
    return res_points

if __name__ == "__main__": 
    
    data_path = 'data'
    
    case_name = 'example'
    if not os.path.exists(case_name):
        os.makedirs(case_name)
        
    measured_depth_input = [7,11,15]
    domain_points = [7,15]
    
    layers_theta = [2, 50, 50, 50, 50, 50, 50, 50, 1]
    layers_coef = [1, 20, 20,  1]
    
    respoints_number = 100
    noise = 0.0
    

    data = pd.read_csv(data_path+'/th.txt',delim_whitespace=True,header=None)
    
    domain_depth = np.linspace(domain_points[0]*0.01,domain_points[1]*0.01,domain_points[1]-domain_points[0]+1)
    domain_time_series = np.array(data)[:,0].T
    domain_idx = (np.int32(domain_depth*100+1)).tolist()
    domain_points = mesh_2d(domain_depth,domain_time_series)
    

    idx = np.int32(np.linspace(0,4999,5000))
    measured_time_series = domain_time_series[idx]
    measured_depth = np.array(measured_depth_input)*0.01
    measured_points = mesh_2d(measured_depth,measured_time_series)
    measured_idx = [points+1 for points in measured_depth_input]
    measured_theta = np.array(data.iloc[:,measured_idx])[idx,:]
    measured_points_train = measured_points
    
    
    theta_train = measured_theta.flatten()[:,np.newaxis]
    

    theta_train = theta_train + noise * np.std(theta_train) * np.random.randn(theta_train.shape[0], theta_train.shape[1])
    

    distribution = chaospy.generate_samples(respoints_number, domain=2, rule='halton')
    res_points = cvt_respoints(distribution = distribution,
                               start_x = np.min(measured_depth),end_x=np.max(measured_depth),
                               start_t=np.min(measured_time_series),end_t=np.max(measured_time_series))
    
    distribution = chaospy.generate_samples(100000, domain=2, rule='halton')
    sparse_points = cvt_respoints(distribution = distribution,
                               start_x = np.min(measured_depth),end_x=np.max(measured_depth),
                               start_t=np.min(measured_time_series),end_t=np.max(measured_time_series))
    
    model = DeepGS(measured_points_train, res_points, theta_train, layers_theta, layers_coef)
    model.pre_train(max_iter=50000,sparse=np.zeros([10,1]))
    
    f_pred = np.stack(model.predict(sparse_points))[:,:,0].T

    x = f_pred[:,1:-1]
    y = f_pred[:,-1:]
    u = f_pred[:,0:1]
    
    Theta_grouped,Uz_grouped,Ut_grouped,norm_x,norm_y = data_group(x, y, u)
    Xi,Tol,AIC_loss = TrainGS(Uz_grouped, Ut_grouped, norm_x,norm_y)
    best_x = Xi[np.argmin(AIC_loss)]
    parsi_terms = np.copy(best_x)
    parsi_terms[:,0][np.nonzero(parsi_terms[:,0])] = 1
    parsi_terms = parsi_terms[:,0][:,np.newaxis]
    sparse=parsi_terms
    
    model = DeepGS(measured_points_train, res_points, theta_train, layers_theta, layers_coef)
    
    for i in range(5):
        model.train(max_iter=100000,sparse=sparse)
        
        f_pred = np.stack(model.predict(sparse_points))[:,:,0].T

        x = f_pred[:,1:-1]
        y = f_pred[:,-1:]
        u = f_pred[:,0:1]
        
        Theta_grouped,Uz_grouped,Ut_grouped,norm_x,norm_y = data_group(x, y, u)
        Xi,Tol,AIC_loss = TrainGS(Uz_grouped, Ut_grouped, norm_x,norm_y)
        best_x = Xi[np.argmin(AIC_loss)]
        parsi_terms = np.copy(best_x)
        parsi_terms[:,0][np.nonzero(parsi_terms[:,0])] = 1
        parsi_terms = parsi_terms[:,0][:,np.newaxis]
        sparse=parsi_terms
        
    model = DeepGS(measured_points_train, res_points, theta_train, layers_theta, layers_coef)
    model.train(max_iter=100000,sparse=sparse)
    f_pred = np.stack(model.predict(sparse_points))[:,:,0].T
    


