<<<<<<< HEAD
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ： SentimentAnalysis
@File    ：BasicModule.py
@Author  ：郑家祥
@Date    ：2021/6/24 13:06 
@Description：BasicModule是对nn.Module的简易封装，提供快速加载和保存模型的接口
'''
import torch
from torch import nn
import time

class BasicModule(nn.Module):
    """
    封装nn.Module，提供load模型和save模型接口
    """

    def __init__(self):
        super(BasicModule, self).__init__()
        self.modelName = str(type(self))

    def load(self, path):
        '''
        加载指定路径的模型
        '''
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        '''
        保存训练的模型到指定路径
        '''
        if name is None:
            prepath = 'models/' + self.modelName + '_'
            name = time.strftime(prepath + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        print("保存的模型路径为：", name)
=======
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ： SentimentAnalysis
@File    ：BasicModule.py
@Author  ：郑家祥
@Date    ：2021/6/24 13:06 
@Description：BasicModule是对nn.Module的简易封装，提供快速加载和保存模型的接口
'''
import torch
from torch import nn
import time

class BasicModule(nn.Module):
    """
    封装nn.Module，提供load模型和save模型接口
    """

    def __init__(self):
        super(BasicModule, self).__init__()
        self.modelName = str(type(self))

    def load(self, path):
        '''
        加载指定路径的模型
        '''
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        '''
        保存训练的模型到指定路径
        '''
        if name is None:
            prepath = 'models/' + self.modelName + '_'
            name = time.strftime(prepath + '%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(), name)
        print("保存的模型路径为：", name)
>>>>>>> d418ed5a0517a1c34a75d286fd5c685a031f6eb6
        return name