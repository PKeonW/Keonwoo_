import gym
import envs
import numpy as np
import  matplotlib.pyplot as plt
import pandas as pd
from envs.custom_env_dir.dqn_agent import DQNAgent
from envs.custom_env_dir.data_handler import DataHandler
from envs.custom_env_dir.sup_model import SupModel
from envs.custom_env_dir.utils import make_env
from datetime import datetime
from sklearn import neural_network as NN
import pickle

from envs.custom_env_dir.ddpg_agent import DDPGAgent

import os
import time

''' THIS FUNCTION WILL BE CALLED FROM THE MAIN-METHOD BELOW'''

def run_model(optimizer, gamma, lr, replace, hl, k, store_dir, mlp_hl, mlp_af, mlp_sl, input_dims, mlp, knn, n_BM,
              uncontrolled, obs, store_results, development, test, test_final , learner_type):
    # Get collection of train, test, dev sets
    train_collection, dev_collection, test_collection, train_count, dev_count, test_count, full_collection = DataHandler().get_data_7d_3split(
        include_weekends=True, \
        n_episodes=450, start_year=2018, start_month=10, start_day=1)
    # LSTM을 여기다가 대입하자 -> 여기서 전체 모델 학습 / pkl로 모델 저장.

    # print(train_collection)
    # Define EV battery capacity in kWh
    battery_capacity = 24
    # Define residential EV charging rate in kW
    charging_rate = 8
    # Set penalty coefficient for incomplete charging
    penalty_coefficient = 12

    # 하루에 돌 수 있는 총 route의 횟수
    day_route =4
    n_nodes = 4

    # Get current directory to store model
    cwd = os.getcwd()

    # Initialize best_score for tracking best model
    best_score = -np.inf

    # Set parameters for testing
    if test:
        # Use development set for parameter tuning
        if development:
            test_collection = dev_collection
            dataset = 'DEV'

        else:
            dataset = 'TEST'

        # Create test environment and pass collection of days
        #ChargingEsssoc-v0
        #ChargingEnv-v0
        env = gym.make('Dynamic_-v0', game_collection=test_collection,
                       battery_capacity=battery_capacity, charging_rate=charging_rate,
                       penalty_coefficient=penalty_coefficient, obs=obs)
        # Simulate each day in the set 10 times with different driving profiles
        n_episodes = len(test_collection) * 10
        # Makes sure the agent does not learn during training
        pre_training_steps = np.inf
        # Load previously trained model
        load_checkpoint = True
        filename = dataset + '_' + optimizer + '_gamma' + str(gamma) + '_lr' + (
            ('%.15f' % lr).rstrip('0').rstrip('.')) + '_replace' + str(replace) + '_HL' + str(hl) +str(learner_type)

        # Set parameters for training
    else:
        # Create training environment and pass collection of days
        env = gym.make('Dynamic_-v0', game_collection=train_collection,
                       n_nodes=n_nodes, n_route=day_route, obs=obs)
        # Do not load a checkpoint - train new model
        load_checkpoint = False

        # Train model for n_episodes episodes
        # n_episodes = 50000
        n_episodes = 30000
        # Specify number of random episodes before epsilon starts to decrease
        #pre_training_steps = 5000
        pre_training_steps = 500

        print(
            'Train model for ' + str(n_episodes) + ' episodes with ' + str(pre_training_steps) + ' pre-train steps ...')
        filename = 'TRAIN' + '_' + optimizer + '_gamma' + str(gamma) + '_lr' + (
            ('%.15f' % lr).rstrip('0').rstrip('.')) + '_replace' + str(replace) + '_HL' + str(hl)

    # Print information if using night benchmark
    if n_BM:
        if development:
            print('Night benchmark on development set')
        else:
            print('Night benchmark on test set')
    # Print information if using uncontrolled charging
    if uncontrolled:
        if development:
            print('Uncontrolled benchmark on development set')
        else:
            print('Uncontrolled benchmark on test set')

    # Create the RL agent with a DQN
    agent = DQNAgent(gamma=gamma, fc1_dims=hl[0], fc2_dims=hl[1], epsilon=1.0, lr=lr,
                     input_dims=input_dims, n_actions=len(env.action_space), mem_size=100000,
                     eps_min=0.1, batch_size=32, replace=replace, eps_dec=1e-5, optimizer=optimizer,
                     chkpt_dir=store_dir, algo='DQNAgent', env_name='ChargingEsssoc-v0')


    # agent = DDPGAgent(gamma=gamma, fc1_dims=hl[0], fc2_dims=hl[1], epsilon=1.0, lr=lr,
    #                  input_dims=input_dims, n_actions=len(env.action_space), mem_size=100000,
    #                  eps_min=0.1, batch_size=32, replace=replace, eps_dec=1e-5, optimizer=optimizer,
    #                  chkpt_dir=store_dir, algo='DQNAgent', env_name='ChargingEnv-v0')

    # Load agent/model parameters from previously trained model
    if load_checkpoint:
        if test_final:
            agent.load_models_final()
            filename = filename + '_finalmodel'
        elif mlp:
            sup_model = SupModel().load_model_mlp(store_dir, mlp_hl, mlp_af, mlp_sl)
            sup_scaler = SupModel().load_scaler(store_dir)
            filename = 'MLP_'
        elif knn:
            sup_model = SupModel().load_model_kneighbors(store_dir, k)
            sup_scaler = SupModel().load_scaler(store_dir)

            filename = 'KNN_'
        elif n_BM:
            filename = dataset + '_BENCHMARK_Night_2-6'
        elif uncontrolled:
            filename = dataset + '_BENCHMARK_Uncontrolled'
        else:
            agent.load_models()
        # Do not take any random actions, strictly act according to policy
        agent.epsilon = 0

    # n_steps = 0
    route_tsp = 0

    # Lists to store all relevant data while training or testing
    price_list, ev_soc_list, action_list, dates, day_cats, starts, ends, scores, avg_scores, eps_history, pen_history, steps_array, final_ev_soc, \
    discounted_action_list, temp_list , avg_20scores ,ess_soc_list ,pv_list= [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],[],[],[]

    for i in range(n_episodes):  # n_episodes
        # Get initial observation from the environment
        observation = env.reset(test, i)

        # print(env.hourly_prices['Spot'][145:169])

        # Here, the return is called 'score'
        score = 0
        # Create lists to store training/test data
        episode_prices, episode_ev_soc, episode_actions, capacity_list, episode_discounted_actions, episode_temps,episode_ess_soc ,episode_pv = [], [], [], [], [], [],[],[]

        for n_route in range(day_route):

        # Loop for 24 h/steps in each episode/game
            for n_steps in range(env.n_nodes):




                # Test with night benchmarking approach: immediately discharge the vehicle in the evening, only charge between 02:00-06:00
                if n_BM:

                    if n_steps < 14 and env.ev_soc != 0:
                        action = 1  # Discharge the vehicle before 02:00 a.m.
                    elif n_steps < 14 and env.ev_soc == 0:
                        action = 2  # Do nothing if the vehicle is fully discharged before 2 a.m.
                    elif n_steps < 18 and env.ev_soc != 1:
                        action = 0  # Charge the vehicle between 2-6 a.m.
                    else:
                        action = 2  # Do nothing after vehicle is fully charged

                # Test with simple benchmarking approach: always charge the vehilce (no control mechanism and V2G/V2H)
                elif uncontrolled: # greedy heuristic

                    if env.ev_soc < 1.0 :
                        action = 0
                    else :
                        action = 4

                # test with supervised model
                elif mlp or knn:
                    obs = np.array(observation)
                    # Scale data
                    obs = sup_scaler.transform(obs.reshape(1, -1))
                    # Predict optimal action based on observation
                    action = sup_model.predict(obs.reshape(1, -1))[0]

                # Deep reinforcement agent takes action according to policy learned
                else:
                    action = agent.choose_action(observation)

                # Store each action taken for evaluation and visualization
                episode_actions.append(env.action_space[action])
                episode_discounted_actions.append(env.discounted_action)

                # Take a step and receive reward and new observation

                observation_, reward, done, info = env.step(action,capacity_list)

                # print('parking_step', n_steps)
                # print('observation_, reward', observation_, reward)
                capacity_list.append(info)

                score += reward

                # Fill replay memory while training
                if not load_checkpoint:
                    agent.store_transition(observation, action, reward, observation_)

                    # Start learning after defined number of random steps
                    # 여기서 5000보다 작게 한 500? 으로 해서 그래프 그려보자 , 3만 episode로 하고
                    if i > pre_training_steps:
                        if learner_type =='DQN':
                            # DQN
                            agent.learn()
                        elif learner_type =='DDQN':
                            # DDQN
                            agent.learn_DDQN()

                            # answer sheet
                            #agent.learn_DDQN_ans()


                # Store all prices, temps, ev_soc for each episode
                episode_prices.append(env.hourly_prices['Spot'][n_steps + 168]/ 600)
                episode_pv.append(env.hourly_prices['sun'][n_steps + 168])
                episode_temps.append(env.hourly_prices['temp'][n_steps + 168])




                # Update observation
                observation = observation_


        # Store all relevant data for evaluation


        ess_soc_list.append(episode_ess_soc)
        action_list.append(episode_actions)
        discounted_action_list.append(episode_discounted_actions)


        scores.append(score)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)


        avg_20score = np.mean(scores[-20:])
        avg_20scores.append(avg_20score)
        # Print average score every 100 episodes
        if i % 100 == 0:
            print('episode: ', i, 'score: ', score,
                  ' average score %.1f' % avg_score, 'best score %.2f' % best_score, 'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

        # if i % 1000 == 0:
        #
        #     print('action_list',action_list[-1])
        #     print('pv_list',pv_list[-1])
        #     print('ev_soc_list',ev_soc_list[-1])
        #     print('ess_soc_list',ess_soc_list[-1])
            # print('discounted_action_list',discounted_action_list)

        # Store model parameters when new moving average score outperforms previous best model
        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

    # plt.plot(avg_scores)
    # plt.xlabel("Episode")
    # plt.ylabel("Episode Reward (Smoothed)")
    # plt.title("Episode Reward over Time (Smoothed over window size {})".format(100))
    # plt.savefig("Episode Reward over Time (Smoothed over window size {})".format(100))
    # plt.show()

    # Store final model parameters after all training episodes
    if not load_checkpoint:
        agent.save_models_final()

    # Calculate the average score
    score = sum(scores) / len(scores)

    print('The average score is ', sum(scores) / len(scores))

    # Store data for all training episodes in csv file
    # if store_results:
    #     DataHandler().store_results(price_list, ev_soc_list, action_list, dates, \
    #                                 day_cats, starts, ends, scores, \
    #                                 avg_scores, final_ev_soc, eps_history, pen_history, \
    #                                 filename, optimizer, gamma, lr, replace, store_dir, \
    #                                 discounted_action_list, temp_list)
    # else:
    #     print('Store results disabled')
    last_100_trained_avg = sum(scores[-100:]) / len(scores[-100:])
    print('last_10000_trained_avg',last_100_trained_avg)
    return avg_scores, avg_20scores, sum(scores) / len(scores) ,action_list ,discounted_action_list , ev_soc_list


if __name__ == '__main__':
    i = 1
    cwd = os.getcwd()

    ''' SET INPUT FEATURES WITH STRING '''
    # The obs string and the resepctive input features variables must have been defined in charging_env.py
    # obs = 'obs50(t_step,ev_soc,p-23-p,pv-pv+23)'

    obs = 'DynamicObs(6)'
    # Set respective input dimensions for the DQN
    input_dims = 6

    ''' SET DQN AGENT PARAMETERS '''
    # If you want to test an agent that has been trained already the parameters have to match!
    optimizer = 'Adam'
    gamma = 0.8
    lr = 0.0001
    replace = 2
    hl = [64, 64]

    ''' SELECT AT LEAST ONE OF THE THREE FOLLOWING OPTIONS '''
    # Select dataset for training or test on test/dev set
    do_train = True
    do_dev = False
    do_test = False

    ''' SELECT NO MORE THAN ONE OF THE FOUR FOLLOWING OPTIONS '''
    # Select true if you want to test a previously trained k-NN classifer
    knn = False
    # Select true if you want to test a previously trained MLP classifer
    mlp = False
    # Select if you want to test the night benchmark
    n_BM = False
    # Select if you want to test uncontrolled charging
    uncontrolled = False

    ''' IF k-NN OR MLP SELECTED -> SPECIFY PARAMETERS '''
    # Specifiy k-nearest neighbors parameters if required
    k = 15
    # Specify MLP parameters if required
    mlp_hl = (16)
    mlp_af = 'relu'
    mlp_sl = 'adam'

    ''' SELECT IF YOU WANT TO STORE TRAIN/TEST INFORMATION IN A CSV FILE '''
    store_results = True

    ''' NO ADJUSTMENTS REQUIRED FROM HERE '''
    # Set store directory depending on previous decision
    if knn:
        info = ' | k = ' + str(k) + ' | ' + obs
        store_dir = cwd + '/knn_models/' + 'KNeighbors_k(' + str(k) + ')' + '_' + obs
    elif mlp:
        info = ' | optimizer=' + mlp_sl + ' | activation function: ' + mlp_af + ' | hl: ' + str(mlp_hl) + ' | ' + obs
        store_dir = cwd + '/mlp_models/' + 'MLP_hl(' + str(mlp_hl) + ')_af(' + str(mlp_af) + ')_sl(' + str(
            mlp_sl) + ')' + '_' + obs
    elif n_BM or uncontrolled:
        info = ' Rule-based benchmark specified above '
        store_dir = cwd
    else:
        info = ' | optimizer=' + optimizer + ' | gamma=' + str(gamma) + ' | lr=' + (
            ('%.15f' % lr).rstrip('0').rstrip('.')) + ' | replace=' + str(replace) + ' | HL: ' + str(hl) + ' | ' + obs
        store_dir = cwd + '/dqn_models/' + optimizer + '_gamma' + str(gamma) + '_lr' + (
            ('%.15f' % lr).rstrip('0').rstrip('.')) + '_replace' + str(replace) + '_HL' + str(hl) + '_' + obs

    # Train

    if do_train:
        print('---------- TRAIN session: ', i, info)
        #os.makedirs(store_dir)
        date_now = datetime.today().strftime("%m%d%H%M")
        start = time.time()
        DQN_avg_scores_plt ,DQN_Scores, DQN_avg_score,action_list,discounted_action_list, ev_soc_list = run_model(optimizer, gamma, lr, replace, hl, k, store_dir, mlp_hl, mlp_af, mlp_sl, input_dims, mlp, knn, n_BM,
                       uncontrolled, obs, store_results, development=False, test=False, test_final=False, learner_type='DQN')
        # end = time.time()
        # DQN_time = end - start
        # print('Training DQN took ', DQN_time, ' seconds...')

        start = time.time()
        DDQN_avg_scores_plt,DDQN_Scores , DDQN_avg_score,action_list ,discounted_action_list, ev_soc_list = run_model(optimizer, gamma, lr, replace, hl, k, store_dir, mlp_hl, mlp_af,
                                                  mlp_sl, input_dims, mlp, knn, n_BM,
                                                  uncontrolled, obs, store_results, development=False, test=False,
                                                  test_final=False, learner_type='DDQN')

        # n_BM = True
        # uncontrolled = False
        #
        # n_BM_avg_scores_plt, n_BM_Scores, n_BM_avg_score, action_list, discounted_action_list, ev_soc_list = run_model(
        #     optimizer, gamma, lr, replace, hl, k, store_dir, mlp_hl, mlp_af,
        #     mlp_sl, input_dims, mlp, knn, n_BM ,
        #     uncontrolled, obs, store_results, development=False, test=False,
        #     test_final=False, learner_type='DDQN')
        #
        # n_BM = False
        # uncontrolled = True
        #
        # uncontrolled_plt, uncontrolled_Scores, uncontrolled_score, action_list, discounted_action_list, ev_soc_list = run_model(
        #     optimizer, gamma, lr, replace, hl, k, store_dir, mlp_hl, mlp_af,
        #     mlp_sl, input_dims, mlp, knn, n_BM,
        #     uncontrolled, obs, store_results, development=False, test=False,
        #     test_final=False, learner_type='DDQN')

        # Check if battery degradation cost effects the number of actions
        # action_count = 0
        # disc_action_count = 0
        # action_len = 0
        # disc_action_len = 0
        # for k in action_list:
        #     for i in k:
        #         action_len += 1
        #         if (i != 0) and (i !='-'):
        #             action_count += 1
        #
        # for k in discounted_action_list:
        #     for i in k:
        #         disc_action_len += 1
        #         if (i != 0) and (i !='-'):
        #             disc_action_count += 1
        #
        # print('len(action_list) : ', action_len)
        # print('len(discounted_action_list) : ', disc_action_len)
        # print('number of total actions : ', action_len + disc_action_len)
        #
        # print('number of charging actions : ', action_count)
        # print('number of discounted charging actions : ',disc_action_count)
        # print('number of total charging actions : ',action_count + disc_action_count)
        #
        # print('Show percentage of charging actions : ',(action_count+ disc_action_count) / (action_len+disc_action_len) )
        #
        # end = time.time()
        # DDQN_time = end - start
        # print('Training DDQN took ', DDQN_time, ' seconds...')

        # plt.plot(DQN_Scores  ,alpha=0.2 ,color='pink')
        plt.plot(DQN_avg_scores_plt , label='DQN_avg',color='lightpink')
        plt.xlabel("Episode")
        plt.ylabel("Episode Reward (Smoothed)")
        plt.title("Dynamic flexible capacity - Episode Reward over Time")
        plt.savefig("Episode Reward over Time (Smoothed over window size {})".format(100))
        plt.legend(loc='best')
        plt.show()
        plt.close()


        # plt.plot(DDQN_Scores ,alpha=0.2,color='purple')
        plt.plot(DDQN_avg_scores_plt, label='DDQN_avg',color='violet')
        #plt.plot(n_BM_avg_scores_plt , label ='Heuristic cheap',color='blue')
        #plt.plot( uncontrolled_plt ,label ='Heuristic greedy',color='orange' )
        plt.xlabel("Episode")
        plt.ylabel("Episode Reward (Smoothed)")
        plt.title("Dynamic flexible capacity - Episode Reward over Time")
        plt.savefig("Episode Reward over Time (Smoothed over window size {},datetime{}).png".format((100),date_now))
        plt.legend(loc='best')
        plt.show()
        plt.close()

        # plt.plot(DQN_Scores, alpha=0.2, color='pink')
        # plt.plot(DQN_avg_scores_plt, label='DQN_avg', color='lightpink')
        # plt.plot(DDQN_Scores, alpha=0.2, color='purple')
        # plt.plot(DDQN_avg_scores_plt, label='DDQN_avg', color='violet')
        # plt.xlabel("Episode")
        # plt.ylabel("Episode Reward (Smoothed)")
        # plt.title("Episode Reward over Time (Smoothed over window size {})".format(100))
        # plt.savefig("Episode Reward over Time (Smoothed over window size {})".format(100))
        # plt.legend(loc='best')
        # plt.show()
        # plt.close()

        end = time.time()
        #print('Training DDQN took ', DQN_time, ' seconds...')
        #print('Training DDQN took ', DDQN_time, ' seconds...')


    # DEV
    if do_dev:
        print('---------- DEV session: ' + str(i) + info)
        run_model(optimizer, gamma, lr, replace, hl, k, store_dir, mlp_hl, mlp_af, mlp_sl, input_dims, mlp, knn, n_BM,
                  uncontrolled, obs, store_results, development=True, test=True, test_final=False,learner_type='DDQN')

    # Test
    if do_test:
        print('---------- TEST session: ', i, info)
        DDQN_avg_scores_plt,DDQN_Scores ,DDQN_avg_score,action_list ,discounted_action_list , ev_soc_list = run_model(optimizer, gamma, lr, replace, hl, k, store_dir, mlp_hl, mlp_af, mlp_sl, input_dims, mlp, knn, n_BM,
                  uncontrolled, obs, store_results, development=False, test=True, test_final=False,learner_type='DDQN')

        action_count = 0
        disc_action_count = 0
        action_len = 0
        disc_action_len = 0
        print('TEST 데이터 결과 출력')
        for index in range(len(action_list)):
            print(index, '번째 EV의 ev_soc : ', ev_soc_list[index])
            print(index, '번째 EV방문', '결과 : ', action_list[index] )
            print(index, '번째 DV방문', '결과 : ', discounted_action_list[index])



        for index,value in enumerate(action_list):
            print(index, '번째 EV방문', '결과 : ', value)
            for i in value:
                action_len += 1
                if (i != 0) and (i !='-'):
                    action_count += 1

        # print('action_list',action_list)


        for index,value in enumerate(discounted_action_list):
            print(index, '번째 discounted EV방문', '결과 : ', value)
            for i in value :
                disc_action_len += 1
                if (i != 0) and (i !='-'):
                    disc_action_count += 1

        print('discounted_action_list', discounted_action_list)


        print('len(action_list) : ', action_len)
        print('len(discounted_action_list) : ', disc_action_len)
        print('number of total actions : ', action_len + disc_action_len)

        print('number of charging actions : ', action_count)
        print('number of discounted charging actions : ', disc_action_count)
        print('number of total charging actions : ', action_count + disc_action_count)

        print('Show percentage of charging actions : ',
              (action_count + disc_action_count) / (action_len + disc_action_len))



    # Print information that nothing is trained or tested...
    if not (do_train or do_dev or do_test):
        print('Select do_train / do_dev / do_test')