__author__ = 'Coco'

from SurvivalAnalysis import SurvivalAnalysis
import numpy as np
import random
import scipy.io as sio
import os
import tensorflow as tf

def cost_func (params) :
        session = tf.InteractiveSession()
        """ This function is to the cost function to pass to Bayesian Optimization.
        :type params: ndarray 
        :param params: vector containing network parameters #layers, #hidden units, dropout rate, and random seed
        
        """
        shuffle = int(params[3])
        random.seed(shuffle)
        num_steps=50
        mat_file_path = 'data/KIPAN_Integ.mat'
        p = os.path.join(os.getcwd(), mat_file_path)
        Data = sio.loadmat(p)

        C = np.asarray([c[0] for c in Data['Censored']])
        survival = np.asarray([t[0] for t in Data['Survival']])

        T = np.asarray([t[0] for t in Data['Survival']])
        X = Data['Integ_X']
        index = np.arange(X.shape[0])
        random.shuffle(index)

        X = X[index, :]
        C = C[index]
        T = T[index]
        

        #foldsize denotes the amount of data used for testing. The same amount 
        #of data is used for model selection. The rest is used for training.
        fold = int( len(X) / 10)
        train_set = {}
        test_set = {}
        final_set = {}

        #caclulate the risk group for every patient i: patients who die after i
        sa = SurvivalAnalysis()    
        train_set['X'], train_set['T'], train_set['C'], train_set['A'] = sa.calc_at_risk(X[0:fold * 6,], T[0:fold * 6], C[0:fold * 6]);
        test_set['X'], test_set['T'], test_set['C'], test_set['A'] = sa.calc_at_risk(X[fold * 6: fold * 8,], T[fold * 6: fold * 8], C[fold * 6: fold * 8]);

        # initialization
        n_obs = train_set['X'].shape[0] 
        n_in = train_set['X'].shape[1] 

        test_obs = test_set['X'].shape[0]
        test_in = test_set['X'].shape[1] 

        n_out = 1

        # tensorflow implementation
        def cumsum(x, observations):
                x = tf.reshape(x, (1, observations))
                values = tf.split(1, x.get_shape()[1], x)
                out = []
                prev = tf.zeros_like(values[0])
                for val in values:
                        s = prev + val
                        out.append(s)
                        prev = s
                cumsum = tf.concat(1, out)
                cumsum = tf.reshape(cumsum, (observations, 1))
                return cumsum


        nl = int(params[0]) 
        n_hidden = int(params[1]) 
        do_rate = params[2] 
        prefix = 'results/' + str(nl)+'-'+str(n_hidden)+'-'+str(do_rate) 
	print('cost: ', prefix + '-' + str(shuffle))
        ## data
        input = tf.placeholder(tf.float32, [n_obs, n_in])
        at_risk = tf.placeholder(tf.int32, [n_obs, ])
        observed = tf.placeholder(tf.float32, [n_obs, ])
       
        ## dropout
        keep_prob = tf.placeholder(dtype=tf.float32)
            
        # testing data
        test_input = tf.placeholder(tf.float32, [test_obs, test_in])
        test_at_risk = tf.placeholder(tf.int32, [test_obs, ])
        test_observed = tf.placeholder(tf.float32, [test_obs, ])

        w_6 = tf.Variable(tf.truncated_normal([n_hidden, n_out], dtype=tf.float32)/20)
        
        ## layer_1
        w_1 = tf.Variable(tf.truncated_normal([n_in, n_hidden], dtype=tf.float32)/20)
        output_layer1 = tf.nn.relu(tf.matmul(input, w_1))
        output_layer1_drop = tf.nn.dropout(output_layer1, keep_prob)
        test_output_layer1 = tf.nn.relu(tf.matmul(test_input, w_1))
        ## output layer
        output = tf.matmul(output_layer1_drop, w_6)
        test_output = tf.matmul(test_output_layer1, w_6)

        if nl > 1:
        ## layer_2
            w_2 = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], dtype=tf.float32)/20)
            output_layer2 = tf.nn.relu(tf.matmul(output_layer1_drop, w_2))
            output_layer2_drop = tf.nn.dropout(output_layer2, keep_prob)
            test_output_layer2 = tf.nn.relu(tf.matmul(test_output_layer1, w_2))
            ## output layer
            output = tf.matmul(output_layer2_drop, w_6)
            test_output = tf.matmul(test_output_layer2, w_6)
        if nl > 2:    
        ## layer_3
            w_3 = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], dtype=tf.float32)/20)
            output_layer3 = tf.nn.relu(tf.matmul(output_layer2_drop, w_3))
            output_layer3_drop = tf.nn.dropout(output_layer3, keep_prob)
            test_output_layer3 = tf.nn.relu(tf.matmul(test_output_layer2, w_3))
            ## output layer
            output = tf.matmul(output_layer3_drop, w_6)
            test_output = tf.matmul(test_output_layer3, w_6)
        if nl > 3:
        ## layer_4
            w_4 = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], dtype=tf.float32)/20)
            output_layer4 = tf.nn.relu(tf.matmul(output_layer3_drop, w_4))
            output_layer4_drop = tf.nn.dropout(output_layer4, keep_prob)
            test_output_layer4 = tf.nn.relu(tf.matmul(test_output_layer3, w_4))
            ## output layer
            output = tf.matmul(output_layer4_drop, w_6)
            test_output = tf.matmul(test_output_layer4, w_6)
        if nl > 4:
        # layer_5
            w_5 = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], dtype=tf.float32)/20)
            output_layer5 = tf.nn.relu(tf.matmul(output_layer4_drop, w_5))
            output_layer5_drop = tf.nn.dropout(output_layer5, keep_prob)
            test_output_layer5 = tf.nn.relu(tf.matmul(test_output_layer4, w_5))
            ## output layer
            output = tf.matmul(output_layer5_drop, w_6)
            test_output = tf.matmul(test_output_layer5, w_6)
        
        exp = tf.reverse(tf.exp(output), dims = [True, False])
        partial_sum_a = cumsum(exp, n_obs)
        partial_sum = tf.reverse(partial_sum_a, dims = [True, False]) + 1
        log_at_risk = tf.log(tf.gather(partial_sum, tf.reshape(at_risk, [-1])) + 1e-50)
        diff = tf.sub(output,log_at_risk)
        times = tf.reshape(diff, [-1]) * observed
        cost = - (tf.reduce_sum(times))

        ### test
        test_exp = tf.reverse(tf.exp(test_output), dims = [True, False])
        test_partial_sum_a = cumsum(test_exp, test_obs)
        test_partial_sum = tf.reverse(test_partial_sum_a, dims = [True, False]) + 1
        test_log_at_risk = tf.log(tf.gather(test_partial_sum, tf.reshape(test_at_risk, [-1])) + 1e-50)
        test_diff = tf.sub(test_output,test_log_at_risk)
        test_times = tf.reshape(test_diff, [-1]) * test_observed
        test_cost = - (tf.reduce_sum(test_times))


        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.0001
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.989, staircase=True)

        session.run(tf.initialize_all_variables())
        # optimizer
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

        #tf.initialize_all_variables().run()
        for step in range(num_steps):
            feed_dict = {input:train_set['X'], at_risk:train_set['A'], observed:1-train_set['C'], test_input:test_set['X'], test_at_risk:test_set['A'], test_observed:1-test_set['C'], keep_prob:1-do_rate}

            outputV, test_outputV, costV,_ = session.run([output, test_output, cost, train_step],feed_dict=feed_dict)
            train_c_index = sa.c_index(outputV, train_set['T'], train_set['C'])
            test_c_index = sa.c_index(test_outputV, test_set['T'], test_set['C'])
#            if (step % 10 == 1) :
#                    print("step: " + str(step) + ", cost: " + str(costV) + ", train cIndex: " + str(train_c_index) + ", test cIndex: " + str(test_c_index))
        return 1 - test_c_index

if __name__ == '__main__':
    result = cost_func([1,100, .5, 0.5])
    print result
