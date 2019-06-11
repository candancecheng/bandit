"""
    1、贪心算法
    2、thomoson sampling
    3、UCB
    4.epsilon greedy算法
"""
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle

#显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


sum_every_rewards_greedy = []
sum_every_rewards_TS = []
sum_every_rewards_epsilon = []
sum_every_rewards_ucb = []
class AllAlgorithm:
    #参数k是臂数目
    def __init__(self,k):
        self.k = k
        self.every_act_times = np.zeros(k)
        self.every_rewards = np.zeros(k)
        self.sum_every_rewards = np.zeros(k)
        self.every_mean = np.zeros(k)
        self.first_nor = np.zeros(k)
        self.first_nor = np.random.normal(loc=0.0, scale=1.0, size=self.k)
    def Rewards(self,best_num):
        for i in range(self.k):
            self.every_rewards[i] = np.random.normal(loc = self.first_nor[i],scale = 1.0,size = 1)
            print(self.every_rewards[i])
        #以上是生成双重正态分布
        self.every_act_times[best_num] += 1 #执行动作的次数加1
        print(self.every_act_times)
        self.sum_every_rewards[best_num] += self.every_rewards[best_num] #更新每个动作的累计奖励
        return self.every_rewards
        #return self.sum_every_rewards[best_num]#用于画曲线


    def Train(self,number_to_explore):
        self.first_nor = np.random.normal(loc=0.0, scale=1.0, size=self.k)
        for i in range(self.k):
            self.every_rewards[i] = np.random.normal(loc=self.first_nor[i], scale=1.0, size=1)
            print(self.every_rewards[i])
        # 以上是生成双重正态分布
        for m in range(self.k):
            for j in self.every_act_times:#为什么只遍历一次？？？
                if (j < number_to_explore): #每个臂的测试次数不小于number_to_explore
                    least_explored = np.argmin(self.every_act_times)
                    self.every_act_times[least_explored] += 1  # 执行动作的次数加1
                    print(self.every_act_times)
                    self.sum_every_rewards[least_explored] += self.every_rewards[least_explored]  # 更新每个动作的累计奖励
                    print(self.sum_every_rewards)
                # return self.Rewards(least_explored)
        # 出现的问题，执行一次train返回到reward，因此会结束训练
        # self.every_act_times = np.zeros(self.k)
        # self.sum_every_rewards = np.zeros(self.k)
    def Police(self):
        all_rewards = self.sum_every_rewards.sum()
        if all_rewards > 0:
            self.every_mean = self.sum_every_rewards / all_rewards
        elif all_rewards < 0:
            self.every_mean = -(self.sum_every_rewards / all_rewards)
        else:
            self.every_mean = self.sum_every_rewards
        # print('---------------------------------start1')
        # print(self.sum_every_rewards)
        # print(all_rewards)
        # print(self.every_mean)
        # print('-----------------------------------end1')
    def save_policy(self):
        with open('policy.bin','wb') as f:
            pickle.dump(self.every_mean, f)

    def load_policy(self):
        with open('policy.bin','rb') as f:
            self.every_mean = pickle.load(f)
            # print('---------------------------------start2')
            # print(self.every_mean)
            # print('-----------------------------------end2')



    def GreedyAlgorithm(self,times):
        # 调用该函数时统计执行该动作后获得的奖励并求和，画出对应曲线图
        a = 0
        while(times > 0):
            times -= 1
            best_num = np.argmax(self.every_mean)
            reward = self.Rewards(best_num)
            # print(reward[best_num],"------single")
            a = a + reward[best_num]
            # print(a,'-------a')
            sum_every_rewards_greedy.append(a)
            # print(sum_every_rewards_greedy,"----------here")
        self.DrawTheCurve()

    def ThompsonSampling(self,times):
        #根据每个臂现有的正态分布分布产生一个随机数b，选择所有臂中
        #产生的随机数最大的那个臂去摇
        a = 0
        while(times > 0):
            times -= 1
            for i in range(self.k):
                self.every_rewards[i] = np.random.normal(loc=self.first_nor[i], scale=1.0, size=1)
            best_num = np.argmax(self.every_rewards)
            reward = self.Rewards(best_num)
            a = a + reward[best_num]
            sum_every_rewards_TS.append(a)
        self.DrawTheCurve()
    def UCB(self,times):
        #置信区间上界公式
        a = 0
        while(times > 0):
            times -= 1
            t = self.every_act_times.sum()
            estimated_variances = self.every_mean - self.every_mean ** 2#**2是对里面的每一个元素求平方
            UCB = self.every_mean + np.sqrt(np.minimum(estimated_variances + np.sqrt(2 * np.log(t) / self.every_act_times), 0.25) * np.log(t) / self.every_act_times)
            best_num = np.argmax(UCB)
            reward = self.Rewards(best_num)
            a = a + reward[best_num]
            sum_every_rewards_ucb.append(a)
        self.DrawTheCurve()

    def EpsilonGreedy(self,epsilon,times):
        a = 0
        while(times > 0):
            times -= 1
            if np.random.rand() < epsilon:
                #随机选择所有的臂
                best_num = random.randint(0,(self.k)-1)
                reward = self.Rewards(best_num)
                a = a + reward[best_num]
                sum_every_rewards_epsilon.append(a)
            else:
                # 选择目前收益概率最高的
                best_num = np.argmax(self.every_mean)
                reward = self.Rewards(best_num)
                a = a + reward[best_num]
                sum_every_rewards_epsilon.append(a)
        self.DrawTheCurve()
    def DrawTheCurve(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        x = np.arange(self.k)
        plt.xlabel('摇臂数量')
        plt.ylabel('摇臂对应的执行次数')
        ax1.plot(x,self.every_act_times)#xlable = '第几个臂',ylable = '每个臂的执行次数'
        x = np.arange(self.k)
        plt.xlabel('摇臂数量')
        plt.ylabel('每个臂对应的总奖励')
        ax2.plot(x,self.sum_every_rewards,color = 'red')
        plt.figure(num=2,figsize = (8,5),)
        plt.xlabel('实验次数')
        plt.ylabel('总奖励')

        print(sum_every_rewards_greedy,'----------image1')
        x = np.arange(len(sum_every_rewards_greedy))
        plt.plot(x,sum_every_rewards_greedy,color = 'blue',label = 'greedy')
        print(sum_every_rewards_TS, '----------image2')
        x = np.arange(len(sum_every_rewards_TS))
        plt.plot(x,sum_every_rewards_TS,color = 'green',label = 'TS')
        print(sum_every_rewards_epsilon, '----------image3')
        x = np.arange(len(sum_every_rewards_epsilon))
        plt.plot(x,sum_every_rewards_epsilon,color = 'yellow',label = 'e-greedy')
        print(sum_every_rewards_ucb, '----------image4')
        x = np.arange(len(sum_every_rewards_ucb))
        plt.plot(x,sum_every_rewards_ucb,color = 'black',label = 'UCB')#xlable ='第几个臂',ylable = '每个臂的总奖励',
        # plt.legend(handles = [L1,L2,L3,L4],labels = ['greedy','TS','e-greedy','ucb'],loc = 'best')
        plt.show()

if __name__ == "__main__":
    algo = AllAlgorithm(10)
    algo.Train(10000)
    algo.Police()
    algo.save_policy()
    algo.load_policy()
    algo.GreedyAlgorithm(20)
    algo.ThompsonSampling(20)
    algo.EpsilonGreedy(0.1,20)
    algo.UCB(20)

