"""
This file contains the base class of devices and their constraints and costs.
"""
import cvxpy as cp
import pandas as pd


class Device:  # 通用设备基类
    """
    设备基类，包含设备名称、时刻数量、时间间隔、参数、决策变量名称、约束列表、成本函数、决策变量列表、数据读取函数、决策变量定义函数、约束定义函数、成本定义函数等。
    子类需根据不同设备特点，实现get_data、get_variables、get_constraints、get_cost等函数。
    子类需在__init__中调用父类初始化函数，并传入设备特有的参数。
    """

    def __init__(self, name: str, nT: int, dT: float, config: dict, vars_: dict):
        self.name = name
        self.nT = nT  # 时刻数量，单位：个
        self.delta_T = int(96/nT)
        self.dT = dT  # 时间间隔，单位：小时
        self.config = config  # 传入的设备设计参数，如{'P_min': 0, 'tau_on': 1, 'tau_off': 1}
        self.var_names = vars_  # 设备决策变量名称、及其类型，如{'P': 'float', 'H': 'float'}
        self.cost = None  # 设备运行成本，默认为0
        self.variables = dict()  # 设备决策变量列表
        self.constraints = list()  # 设备约束列表
        self.on_off_flag = False  # 是否需要额外添加启停机变量
        self.check_config()  # 检查参数，自适应添加启停机变量，更新var_names中的变量及其类型
        self.initialize()  # 初始化设备数据读取、决策变量列表、约束列表、成本函数

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def initialize(self):
        """
        初始化设备数据读取、决策变量列表、约束列表、成本函数
        """
        self.get_data()  # 读取数据
        self.get_variables()  # 定义决策变量
        self.get_constraints()  # 定义约束列表
        self.get_cost()  # 定义成本函数

    def check_config(self):  # 检查参数，并自适应添加启停机变量，更新var_names中的变量及其类型
        """
        检查参数，并自适应添加启停机变量，更新var_names中的变量及其类型
        """
        if 'tau_on' in self.config.keys():  # 根据传入参数，判断是否有启停机约束，有则添加变量
            self.var_names.update({
                'v': 'boolean',  # 运行状态：1为运行，0为停机
                'u_on': 'boolean',  # 开机flag
                'u_off': 'boolean'  # 关机flag
            })
            self.on_off_flag = True
        else:
            self.on_off_flag = False
        if ('P_min' in self.config.keys()) and (self.config['P_min'] > 0):  # 若有最小出力限制，需要考虑运行状态
            self.var_names.update({'v': 'boolean'})
        else:
            self.config['P_min'] = 0
        return self.config

    def get_data(self):
        """
        读取设备数据
        """
        pass

    def get_variables(self):  # 根据var_names中记录的变量及变量类型，定义决策变量
        """
        根据var_names中记录的变量及变量类型，定义决策变量
        """
        for var_name, type_ in self.var_names.items():
            if type_ == "float":
                self.variables[var_name] = cp.Variable(self.nT)
            elif type_ == "boolean":
                self.variables[var_name] = cp.Variable(self.nT, boolean=True)
            elif type_ == "integer":
                self.variables[var_name] = cp.Variable(self.nT, integer=True)
            else:
                raise ValueError(f"Invalid parameter type: {type_}")

    def get_cost(self):
        """
        定义成本函数，默认为0
        """
        self.cost = 0

    def get_constraints(self):
        """
        定义约束列表，默认为空
        """
        pass

    def add_on_off_constraints(self):  # 添加启停机约束
        """
        添加启停机约束
        """
        self.constraints += [
            self.variables['v'][1:] - self.variables['v'][:-1] == self.variables['u_on'][1:] - self.variables['u_off'][
                                                                                               1:],
            self.variables['u_on'] + self.variables['u_off'] <= 1
        ]
        for h in range(self.nT):
            tau_on = min(self.nT - h, self.config['tau_on'])  # 最小开机时长
            tau_off = min(self.nT - h, self.config['tau_off'])  # 最小停机时长
            self.constraints += [
                cp.sum([self.variables['v'][h + t] for t in range(0, tau_on)]) >= self.variables['u_on'][h] * tau_on,
                cp.sum([(1 - self.variables['v'][h + t]) for t in range(0, tau_off)]) >= self.variables['u_off'][
                    h] * tau_off]


class PowerLoad(Device):  # 电力负荷 todo: 增加可调节负荷参数，如可转移、可削减等
    """
    电力负荷
    """

    def __init__(self, name: str, nT: int, dT: float, config: dict):
        super().__init__(name, nT, dT, config=config, vars_={})

    def get_data(self):
        """
        读取负荷数据，并根据参数修正
        """
        self.config['P_load'] = pd.read_csv(self.config['path'])[str(self.config['index_col'])].values[::self.delta_T] * \
                                self.config[
                                    'magnification factor']  # 读取负荷数据，并根据参数修正

    def get_constraints(self):
        """
        负荷的电出力为负，热出力为0
        """
        self.variables['P'] = - self.config['P_load']  # 负荷的电出力为负
        self.variables['Q'] = 0
        self.variables['H'] = 0


class QLoad(Device):  # 电力无功负荷
    """
    电力无功负荷
    """

    def __init__(self, name: str, nT: int, dT: float, config: dict):
        super().__init__(name, nT, dT, config=config, vars_={})

    def get_data(self):
        """
        读取负荷数据，并根据参数修正
        """
        self.config['Q_load'] = pd.read_csv(self.config['path'])[str(self.config['index_col'])].values[::self.delta_T] * \
                                self.config[
                                    'magnification factor']  # 读取负荷数据，并根据参数修正

    def get_constraints(self):
        """
        负荷的电出力为0，热出力为负
        """
        self.variables['P'] = 0
        self.variables['Q'] = - self.config['Q_load']  # 负荷的电出力为负
        self.variables['H'] = 0


class HeatLoad(Device):  # 热负荷
    """
    热负荷
    """

    def __init__(self, name: str, nT: int, dT: float, config: dict):
        super().__init__(name, nT, dT, config=config, vars_={})

    def get_data(self):
        """
        读取负荷数据，并根据参数修正
        """
        self.config['H_load'] = pd.read_csv(self.config['path'])[str(self.config['index_col'])].values[::self.delta_T] * \
                                self.config[
                                    'magnification factor']  # 读取负荷数据，并根据参数修正

    def get_constraints(self):
        """
        负荷的电出力为0，热出力为负
        """
        self.variables['P'] = 0
        self.variables['Q'] = 0
        self.variables['H'] = - self.config['H_load']  # 负荷的热出力为负


class PV(Device):  # 光伏发电
    """
    光伏发电
    """

    def __init__(self, name: str, nT: int, dT: float, config: dict):
        """
        设备参数：电出力
        """
        super().__init__(name, nT, dT, config=config, vars_={'P': 'float', 'Q': 'float'})

    def get_data(self):
        """
        读取光伏数据，并根据参数修正最大出力
        """
        self.config['P_max'] = pd.read_csv(self.config['path'])[str(self.config['index_col'])].values[::self.delta_T] * \
                               self.config[
                                   'magnification factor']  # 读取光伏数据，并根据参数修正最大出力

    def get_constraints(self):
        """
        光伏的最大出力约束，热出力为0
        """
        self.variables['H'] = 0
        self.constraints += [
            self.variables['P'] >= 0,
            self.variables['P'] <= self.config['P_max'],
            self.variables['Q'] <= self.variables['P'] / 0.9 * (1 - 0.9 ** 2) ** 0.5,
            self.variables['Q'] >= -self.variables['P'] / 0.9 * (1 - 0.9 ** 2) ** 0.5]


class WP(Device):  # 风电
    """
    风电
    """

    def __init__(self, name: str, nT: int, dT: float, config: dict):
        """
        设备参数：电出力
        """
        super().__init__(name, nT, dT, config=config, vars_={'P': 'float', 'Q': 'float'})

    def get_data(self):
        """
        读取风电数据，并根据参数修正最大出力
        """
        self.config['P_max'] = pd.read_csv(self.config['path'])[str(self.config['index_col'])].values[::self.delta_T] * \
                               self.config['magnification factor']  # 读取风电数据，并根据参数修正最大出力

    def get_constraints(self):
        """
        风电的最大出力约束，热出力为0
        """
        self.variables['H'] = 0
        self.constraints += [
            self.variables['P'] >= 0,
            self.variables['P'] <= self.config['P_max'],
            self.variables['Q'] >= -self.variables['P'] / 0.9 * (1 - 0.9 ** 2) ** 0.5,
            self.variables['Q'] <= self.variables['P'] / 0.9 * (1 - 0.9 ** 2) ** 0.5]


class CHP(Device):  # 热电联产
    """
    热电联产
    """

    def __init__(self, name: str, nT: int, dT: float, config: dict):
        """
        设备参数：电出力、热出力
        """
        super().__init__(name, nT, dT, config=config, vars_={
            'P': 'float',
            'H': 'float',
            'Q': 'float'})

    def get_constraints(self):
        """
        热电联产的电出力约束，热出力约束
        """
        self.constraints += [
            self.variables['P'] >= 0,
            self.variables['P'] <= self.config['P_max'],
            self.variables['Q'] >= -self.variables['P'] / 0.8 * (1 - 0.8 ** 2) ** 0.5,
            self.variables['Q'] <= self.variables['P'] / 0.8 * (1 - 0.8 ** 2) ** 0.5
        ]
        self.constraints += [
            self.variables['H'] >= 0,
            # self.variables['H'] >= self.variables['P'],
            self.variables['H'] <= self.variables['P'] * self.config['alpha']
        ]
        if 'v' in self.variables.keys():
            self.constraints += [self.variables['P'] <= self.variables['v'] * self.config['P_max'],
                                 self.variables['P'] >= self.variables['v'] * self.config['P_min']]
        else:
            self.constraints += [self.variables['P'] <= self.config['P_max'],
                                 self.variables['P'] >= self.config['P_min']]
        if self.on_off_flag:
            self.add_on_off_constraints()

    def get_cost(self):
        """
        热电联产的成本函数
        """
        self.cost = cp.sum(self.config['gas_p'] * self.variables['P'] * (1 + self.config['alpha'])) * self.dT
        # self.cost = cp.sum(self.config['gas_p'] * (self.variables['P'] + self.variables['H'])) * self.dT
        # self.cost = cp.sum(self.config['gas_p'] * (self.variables['P'] + self.variables['H']) + self.variables[
        #     'P'] ** 2 * 0.00001) * self.dT
        # self.cost = cp.sum(self.variables['P']) * self.dT


class Battery(Device):  # 电池
    """
    电池
    """

    def __init__(self, name: str, nT: int, dT: float, config: dict):
        """
        设备参数：SOC、充电功率、放电功率
        """
        super().__init__(name, nT, dT, config=config, vars_={
            'E': 'float',
            'P_charge': 'float',
            'P_discharge': 'float',
            'Q_charge': 'float',
            'Q_discharge': 'float',
            'Q': 'float'})

    def get_constraints(self):
        """
        电池的充电、放电、SOC约束
        """
        self.variables['P'] = self.variables['P_discharge'] - self.variables['P_charge']  # 净发电
        self.variables['Q'] = self.variables['Q_discharge'] - self.variables['Q_charge']  # 净用电
        self.variables['H'] = 0
        self.config['P_max'] = self.config['E_max']/3

        self.constraints += [
            self.variables['Q_charge'] >= -self.variables['P_charge'] / 0.8 * (1 - 0.8 ** 2) ** 0.5,
            self.variables['Q_charge'] <= self.variables['P_charge'] / 0.8 * (1 - 0.8 ** 2) ** 0.5,
            self.variables['Q_discharge'] >= -self.variables['P_discharge'] / 0.8 * (1 - 0.8 ** 2) ** 0.5,
            self.variables['Q_discharge'] <= self.variables['P_discharge'] / 0.8 * (1 - 0.8 ** 2) ** 0.5,
            self.variables['P_charge'] <= self.config['P_max'],
            self.variables['P_discharge'] >= 0,
            self.variables['P_discharge'] <= self.config['P_max'],
            self.variables['P_charge'] >= 0,
            self.variables['E'] >= 0,
            self.variables['E'] <= self.config['E_max'],
            self.variables['E'][-1] == self.config['E_max']*0.3,
            self.variables['E'] - cp.hstack([self.config['E_max']*0.3, self.variables['E'][:-1]]) * (
                    1 - self.config['r_loss'] * self.dT) == self.variables['P_charge'] * self.config['eta_charge'] -
            self.variables['P_discharge'] / self.config['eta_discharge']
        ]


class HeatStorage(Device):  # 储热
    """
    储热
    """

    def __init__(self, name: str, nT: int, dT: float, config: dict):
        """
        设备参数：SOC、储热功率、放热功率
        """
        super().__init__(name, nT, dT, config=config, vars_={
            'E': 'float',
            'H_rel': 'float',
            'H_sto': 'float'})

    def get_constraints(self):
        """
        储热的储热、放热、SOC约束
        """
        self.variables['P'] = 0
        self.variables['Q'] = 0
        self.config['H_max'] = self.config['E_max'] / 3
        self.variables['H'] = self.variables['H_rel'] - self.variables['H_sto']  # 净产热
        self.constraints += [
            self.variables['H_rel'] <= self.config['H_max'],
            self.variables['H_rel'] >= 0,
            self.variables['H_sto'] >= 0,
            self.variables['H_sto'] <= self.config['H_max'],
            self.variables['E'] >= 0,
            self.variables['E'] <= self.config['E_max'],
            self.variables['E'][-1] == self.config['E_max']*0.3,
            self.variables['E'] - cp.hstack([self.config['E_max']*0.3, self.variables['E'][:-1]]) * (
                    1 - self.config['r_loss'] * self.dT) == self.variables['H_sto'] - self.variables['H_rel']
        ]


class ElectricHeatPump(Device):  # 电热泵
    """
    电热泵
    """

    def __init__(self, name: str, nT: int, dT: float, config: dict):
        """
        设备参数：电出力、热出力
        """
        super().__init__(name, nT, dT, config=config, vars_={
            'P': 'float',
            'H': 'float',
            'Q': 'float'})

    def get_constraints(self):
        """
        电热泵的电出力约束，热出力约束
        """

        self.constraints += [
            self.variables['H'] == - self.variables['P'] * self.config['cop'],  # 产生热量，为正数
            self.variables['Q'] >= self.variables['P'] / 0.8 * (1 - 0.8 ** 2) ** 0.5,
            self.variables['Q'] <= -self.variables['P'] / 0.8 * (1 - 0.8 ** 2) ** 0.5
        ]
        if 'v' in self.variables.keys():
            self.constraints += [self.variables['P'] >= - self.variables['v'] * self.config['P_max'],
                                 self.variables['P'] <= - self.variables['v'] * self.config['P_min']]
        else:
            self.constraints += [self.variables['P'] >= - self.config['P_max'],
                                 self.variables['P'] <= - self.config['P_min']]
        if self.on_off_flag:
            self.add_on_off_constraints()


class ElectricHeater(Device):  # 电制热/电锅炉
    """
    电制热/电锅炉
    """

    def __init__(self, name: str, nT: int, dT: float, config: dict):
        """
        设备参数：电出力、热出力
        """
        super().__init__(name, nT, dT, config=config, vars_={
            'P': 'float',
            'H': 'float'})

    def get_constraints(self):
        """
        电制热/电锅炉的电出力约束，热出力约束
        """
        self.variables['Q'] = 0
        self.constraints += [
            self.variables['H'] == - self.variables['P'] * self.config['eta']  # 产生热量，为正数
        ]
        if 'v' in self.variables.keys():
            self.constraints += [self.variables['P'] >= - self.variables['v'] * self.config['P_max'],
                                 self.variables['P'] <= - self.variables['v'] * self.config['P_min']]
        else:
            self.constraints += [self.variables['P'] >= - self.config['P_max'],
                                 self.variables['P'] <= - self.config['P_min']]
        if self.on_off_flag:
            self.add_on_off_constraints()

class GridInterface(Device):  # 与电网间接口
    """
    与电网间接口
    """

    def __init__(self, name: str, nT: int, dT: float, config: dict):
        """
        设备参数：购电、售电功率曲线
        """
        super().__init__(name, nT, dT, config=config, vars_={
            'P_buy': 'float',
            'P_sell': 'float',
            'Q_buy': 'float',
            'Q_sell': 'float',
            'Q': 'float'})

    def get_data(self):
        """
        读取购电、售电价格曲线
        """
        self.config['price_buy'] = pd.read_csv(self.config['price_buy']['path'])[
                                       str(self.config['price_buy']['index_col'])].values[::self.delta_T] * \
                                   self.config['price_buy'][
                                       'magnification factor']
        self.config['price_sell'] = pd.read_csv(self.config['price_sell']['path'])[
                                        str(self.config['price_sell']['index_col'])].values[::self.delta_T] * \
                                    self.config['price_sell'][
                                        'magnification factor']

    def get_constraints(self):
        """
        与电网间接口的购电、售电功率约束
        """
        self.variables['P'] = self.variables['P_buy'] - self.variables['P_sell']  # 净用网电
        self.variables['Q'] = self.variables['Q_buy'] - self.variables['Q_sell']  # 净用网电
        self.variables['H'] = 0

        self.constraints += [
            self.variables['P_buy'] >= 0,
            self.variables['P_sell'] >= 0,
            self.variables['P_buy'] <= self.config['P_buy_max'],
            self.variables['P_sell'] <= self.config['P_sell_max'],
            self.variables['Q_buy'] >= -self.variables['P_buy'] / 0.8 * (1 - 0.8 ** 2) ** 0.5,
            self.variables['Q_buy'] <= self.variables['P_buy'] / 0.8 * (1 - 0.8 ** 2) ** 0.5,
            self.variables['Q_sell'] >= -self.variables['P_sell'] / 0.8 * (1 - 0.8 ** 2) ** 0.5,
            self.variables['Q_sell'] <= self.variables['P_sell'] / 0.8 * (1 - 0.8 ** 2) ** 0.5
        ]

    def get_cost(self):
        """
        与电网间接口的购电、售电成本
        购电成本-售电成本
        """
        self.cost = cp.sum(
            cp.multiply(self.config['price_buy'], self.variables['P_buy']) - cp.multiply(self.config['price_sell'],
                                                                                         self.variables[
                                                                                             'P_sell'])) * self.dT
