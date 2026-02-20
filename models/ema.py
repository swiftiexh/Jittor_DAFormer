# EMA 模型封装

import jittor as jt
from copy import deepcopy

class EMAModel:
    def __init__(self, model, alpha=0.999):
        self.alpha = alpha
        self.model = deepcopy(model)

        # 教师模型进入推理模式
        self.model.eval()

        # 冻结教师模型参数,不参与梯度计算
        for param in self.model.parameters():
            param.stop_grad()

    def update(self, iteration, student_model):
        # 计算动态 alpha
        # 训练初期 alpha 小,教师快速适应学生
        # 训练后期 alpha 大,教师更新更平滑
        alpha_teacher = min(1.0 - 1.0 / (iteration + 1), self.alpha)
        # 更新教师模型参数
        with jt.no_grad():
            for ema_param, student_param in zip(self.model.parameters(), student_model.parameters()):
                # Jittor 没有 .data 的概念，Var 本身就可以参与计算和 in-place 更新、
                # 不用区分标量，Jittor 会正确广播
                ema_param.update(
                    alpha_teacher * ema_param + 
                    (1 - alpha_teacher) * student_param
                )
    
    # 获取 EMA 模型的状态字典
    def state_dict(self):
        return self.model.state_dict()
    
    # 加载状态字典
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)