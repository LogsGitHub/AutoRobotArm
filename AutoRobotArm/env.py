import numpy as np
import pyglet


class ArmEnv(object):
    viewer = None  # 定义一个可视化对象，初始为空
    dt = .1  # 刷新率，每0.1秒刷新一次
    action_bound = [-1, 1]  # 动作的范围，-1到1之间
    goal = {'x': 100., 'y': 100., 'l': 40}  # 目标的位置和大小，用一个字典表示
    state_dim = 9  # 状态的维度，有9个特征
    action_dim = 2  # 动作的维度，有2个动作

    def __init__(self):  # 初始化方法
        self.arm_info = np.zeros(  # 定义一个数组，用来存储两个臂的信息
            2, dtype=[('l', np.float32), ('r', np.float32)])
        self.arm_info['l'] = 100  # 两个臂的长度都是100
        self.arm_info['r'] = np.pi / 6  # 两个臂的初始角度都是pi/6
        self.on_goal = 0  # 定义一个变量，用来记录机器人是否达到目标

    def step(self, action):  # 定义一个步进方法，用来更新状态和奖励
        done = False  # 定义一个变量，用来表示是否结束
        action = np.clip(action, *self.action_bound)  # 将动作限制在范围内
        self.arm_info['r'] += action * self.dt  # 根据动作和刷新率更新角度信息
        self.arm_info['r'] %= np.pi * 2  # 将角度归一化到0到2pi之间
        if self.arm_info['r'][1] > np.pi:  # 增加一个判断条件，如果前臂的旋转角度超过了180度
            self.arm_info['r'][1] = np.pi  # 将前臂的旋转角度限制在180度以内
        elif self.arm_info['r'][1] < -np.pi:  # 增加一个判断条件，如果前臂的旋转角度小于-180度
            self.arm_info['r'][1] = -np.pi  # 将前臂的旋转角度限制在-180度以内
        (a1l, a2l) = self.arm_info['l']  # 取出两个臂的长度
        (a1r, a2r) = self.arm_info['r']  # 取出两个臂的角度
        a1xy = np.array([200., 200.])  # 定义第一个臂的起点坐标（200，200）
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # 根据第一个臂的长度和角度计算第一个臂的终点坐标（也是第二个臂的起点坐标）
        finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # 根据两个臂的长度和角度计算第二个臂的终点坐标（也是机器人的手指坐标）
        # 归一化特征
        dist1 = [(self.goal['x'] - a1xy_[0]) / 400, (self.goal['y'] - a1xy_[1]) / 400]  # 计算第一个臂的终点到目标中心的距离，并除以400进行归一化
        dist2 = [(self.goal['x'] - finger[0]) / 400,
                 (self.goal['y'] - finger[1]) / 400]  # 计算第二个臂的终点到目标中心的距离，并除以400进行归一化
        r = -np.sqrt(dist2[0] ** 2 + dist2[1] ** 2)  # 根据第二个臂的终点到目标中心的距离计算奖励值，距离越小奖励越大
        r -= 0.1

        # 判断是否结束和奖励
        if self.goal['x'] - self.goal['l'] / 2 < finger[0] < self.goal['x'] + self.goal[
            'l'] / 2:  # 如果第二个臂的终点的x坐标在目标的范围内
            if self.goal['y'] - self.goal['l'] / 2 < finger[1] < self.goal['y'] + self.goal[
                'l'] / 2:  # 如果第二个臂的终点的y坐标在目标的范围内
                r += 1.5  # 奖励值加1
                self.on_goal += 1  # 记录达到目标的次数加1
                if self.on_goal > 30:  # 如果达到目标的次数超过50次
                    done = True  # 结束为真
                    action = [0., 0.]  # 增加一个终止条件，防止抖动
        else:  # 否则
            self.on_goal = 0  # 记录达到目标的次数归零

        # 状态
        s = np.concatenate(
            (a1xy_ / 200, finger / 200, dist1 + dist2, [1. if self.on_goal else 0.]))  # 将所有特征拼接成一个数组，作为状态
        return s, r, done  # 返回状态，奖励和结束

    def reset(self):  # 定义一个重置方法，用来初始化环境
        self.goal['x'] = np.random.rand() * 400.  # 随机生成目标的x坐标
        self.goal['y'] = np.random.rand() * 400.  # 随机生成目标的y坐标
        self.arm_info['r'] = 2 * np.pi * np.random.rand(2)  # 随机生成两个臂的角度
        self.on_goal = 0  # 记录达到目标的次数归零
        (a1l, a2l) = self.arm_info['l']  # 取出两个臂的长度
        (a1r, a2r) = self.arm_info['r']  # 取出两个臂的角度
        a1xy = np.array([200., 200.])  # 定义第一个臂的起点坐标（200，200）
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # 根据第一个臂的长度和角度计算第一个臂的终点坐标（也是第二个臂的起点坐标）
        finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # 根据两个臂的长度和角度计算第二个臂的终点坐标（也是机器人的手指坐标）
        # 归一化特征
        dist1 = [(self.goal['x'] - a1xy_[0]) / 400, (self.goal['y'] - a1xy_[1]) / 400]  # 计算第一个臂的终点到目标中心的距离，并除以400进行归一化
        dist2 = [(self.goal['x'] - finger[0]) / 400,
                 (self.goal['y'] - finger[1]) / 400]  # 计算第二个臂的终点到目标中心的距离，并除以400进行归一化
        # 状态
        s = np.concatenate(
            (a1xy_ / 200, finger / 200, dist1 + dist2, [1. if self.on_goal else 0.]))  # 将所有特征拼接成一个数组，作为状态
        return s  # 返回状态

    def render(self):  # 定义一个渲染方法，用来显示环境
        if self.viewer is None:  # 如果可视化对象为空
            self.viewer = Viewer(self.arm_info, self.goal)  # 创建一个Viewer对象，传入两个臂的信息和目标信息
        self.viewer.render()  # 调用Viewer对象的渲染方法

    def sample_action(self):  # 定义一个采样动作方法，用来随机生成动作
        return np.random.rand(2) - 0.5  # 返回一个-0.5到0.5之间的随机数组，作为'''


# 定义一个Viewer类，继承自pyglet.window.Window类，用来显示双臂机器人的环境
class Viewer(pyglet.window.Window):
    bar_thc = 5  # 定义一个常量，表示臂的厚度

    def __init__(self, arm_info, goal):  # 初始化方法，传入两个臂的信息和目标信息
        # vsync=False to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(width=400, height=400, resizable=False, caption='Arm',
                                     vsync=False)  # 调用父类的初始化方法，设置窗口的宽度，高度，是否可调整大小，标题和垂直同步
        pyglet.gl.glClearColor(1, 1, 1, 1)  # 设置窗口的背景颜色为白色
        self.arm_info = arm_info  # 将两个臂的信息赋值给实例属性
        self.goal_info = goal  # 将目标信息赋值给实例属性
        self.center_coord = np.array([200, 200])  # 定义一个数组，表示窗口的中心坐标（200，200）

        self.batch = pyglet.graphics.Batch()  # display whole batch at once
        self.goal = self.batch.add(  # 在批处理中添加一个目标对象，用四边形表示
            4, pyglet.gl.GL_QUADS, None,  # 4 corners
            ('v2f', [goal['x'] - goal['l'] / 2, goal['y'] - goal['l'] / 2,  # location
                     goal['x'] - goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] - goal['l'] / 2]),
            ('c3B', (86, 109, 249) * 4))  # color
        self.arm1 = self.batch.add(  # 在批处理中添加一个第一个臂对象，用四边形表示
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250,  # location
                     250, 300,
                     260, 300,
                     260, 250]),
            ('c3B', (249, 86, 86) * 4,))  # color
        self.arm2 = self.batch.add(  # 在批处理中添加一个第二个臂对象，用四边形表示
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,  # location
                     100, 160,
                     200, 160,
                     200, 150]), ('c3B', (249, 86, 86) * 4,))
        (a1l, a2l) = self.arm_info['l']
        self.min_radius = abs(a1l - a2l)
        self.max_radius = a1l + a2l

    def render(self):  # 定义一个渲染方法，用来显示环境
        self._update_arm()  # 调用私有方法更新臂的位置和角度
        self.switch_to()  # 切换到当前窗口
        self.dispatch_events()  # 处理窗口事件
        self.dispatch_event('on_draw')  # 触发绘制事件
        self.flip()  # 刷新窗口

    def on_draw(self):  # 定义一个绘制方法，用来绘制批处理中的对象
        self.clear()  # 清空窗口内容
        self.batch.draw()  # 绘制批处理中的对象

    def _update_arm(self):  # 定义一个私有方法，用来更新臂的位置和角度
        # update goal
        self.goal.vertices = (  # 根据目标信息更新目标对象的顶点坐标
            self.goal_info['x'] - self.goal_info['l'] / 2, self.goal_info['y'] - self.goal_info['l'] / 2,
            self.goal_info['x'] + self.goal_info['l'] / 2, self.goal_info['y'] - self.goal_info['l'] / 2,
            self.goal_info['x'] + self.goal_info['l'] / 2, self.goal_info['y'] + self.goal_info['l'] / 2,
            self.goal_info['x'] - self.goal_info['l'] / 2, self.goal_info['y'] + self.goal_info['l'] / 2)

        # update arm
        (a1l, a2l) = self.arm_info['l']  # 取出两个臂的长度
        (a1r, a2r) = self.arm_info['r']  # 取出两个臂的角度
        a1xy = self.center_coord  # 取出第一个臂的起点坐标（200，200）
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # 根据第一个臂的长度和角度计算第一个臂的终点坐标（也是第二个臂的起点坐标）
        a2xy_ = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # 根据两个臂的长度和角度计算第二个臂的终点坐标（也是机器人的手指坐标）

        a1tr, a2tr = np.pi / 2 - self.arm_info['r'][0], np.pi / 2 - self.arm_info['r'].sum()  # 计算两个臂的垂直角度
        xy01 = a1xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc  # 计算第一个臂对象的左上角顶点坐标
        xy02 = a1xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc  # 计算第一个臂对象的左下角顶点坐标
        xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc  # 计算第一个臂对象的右下角顶点坐标
        xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc  # 计算第一个臂对象的右上角顶点坐标

        xy11_ = a1xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc  # 计算第二个臂对象的左上角顶点坐标
        xy12_ = a1xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc  # 计算第二个臂对象的左下角顶点坐标
        xy21 = a2xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc  # 计算第二个臂对象的右上角顶点坐标
        xy22 = a2xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc  # 计算第二个臂对象的右下角顶点坐标

        self.arm1.vertices = np.concatenate((xy01, xy02, xy11, xy12))  # 根据计算出来的顶点坐标更新第一个臂对象的顶点属性
        self.arm2.vertices = np.concatenate((xy11_, xy12_, xy21, xy22))  # 根据计算出来的顶点坐标更新第二个臂对象的顶点属性

    # convert the mouse coordinate to goal's coordinate
    def on_mouse_motion(self, x, y, dx, dy):  # 定义一个鼠标移动事件处理方法，用来更新目标信息
        dx, dy = x - self.center_coord[0], y - self.center_coord[1]
        distance = np.sqrt(dx ** 2 + dy ** 2)
        if distance < self.min_radius:
            x = self.center_coord[0] + dx * self.min_radius / distance
            y = self.center_coord[1] + dy * self.min_radius / distance
        elif distance > self.max_radius:
            x = self.center_coord[0] + dx * self.max_radius / distance
            y = self.center_coord[1] + dy * self.max_radius / distance
        self.goal_info['x'] = x  # 将鼠标的x坐标赋值给目标信息中的x属性
        self.goal_info['y'] = y


if __name__ == '__main__':
    env = ArmEnv()
    while True:
        env.render()
        env.step(env.sample_action())
