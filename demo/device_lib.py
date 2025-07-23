"""
This file contains the base class of devices and their constraints and costs.
"""
import cvxpy as cp
import pandas as pd


class Device:  # General device base class
    """
    Base class for devices, containing attributes such as device name, number of time steps, time interval, parameters,
    decision variable names, constraint list, cost function, decision variable list, data reading function,
    decision variable definition function, constraint definition function, cost definition function, etc.
    Subclasses need to implement the get_data, get_variables, get_constraints, get_cost functions according to the characteristics of different devices.
    Subclasses need to call the parent class initialization function in __init__ and pass in parameters specific to the device.
    """

    def __init__(self, name: str, nT: int, dT: float, config: dict, vars_: dict):
        self.name = name
        self.nT = nT  # Number of time steps, unit: count
        self.delta_T = int(96 / nT)
        self.dT = dT  # Time interval, unit: hours
        self.config = config  # Device design parameters passed in, e.g., {'P_min': 0, 'tau_on': 1, 'tau_off': 1}
        self.var_names = vars_  # Names and types of device decision variables, e.g., {'P': 'float', 'H': 'float'}
        self.cost = None  # Device operating cost, default is 0
        self.variables = dict()  # List of device decision variables
        self.constraints = list()  # List of device constraints
        self.on_off_flag = False  # Whether to add additional start-up/shutdown variables
        self.check_config()  # Check parameters, adaptively add start-up/shutdown variables, and update variable names and types in var_names
        self.initialize()  # Initialize device data reading, decision variable list, constraint list, and cost function

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def initialize(self):
        """
        Initialize device data reading, decision variable list, constraint list, and cost function
        """
        self.get_data()  # # Read data
        self.get_variables()  # Define decision variables
        self.get_constraints()  # Define the list of constraints
        self.get_cost()  # Define the cost function

    def check_config(self):
        """
        Check parameters, adaptively add start-up/shutdown variables, and update variable names and types in var_names
        """
        if 'tau_on' in self.config.keys():  # Based on the input parameters, determine if there are start-up/shutdown constraints; if so, add corresponding variables.
            self.var_names.update({
                'v': 'boolean',  # Operating status: 1 for running, 0 for shutdown
                'u_on': 'boolean',  # Startup flag
                'u_off': 'boolean'  # Shutdown flag
            })
            self.on_off_flag = True
        else:
            self.on_off_flag = False
        if ('P_min' in self.config.keys()) and (self.config['P_min'] > 0):  # If there is a minimum output limit, the operating status needs to be considered.
            self.var_names.update({'v': 'boolean'})
        else:
            self.config['P_min'] = 0
        return self.config

    def get_data(self):
        """
        Read device data
        """
        pass

    def get_variables(self):
        """
        Define decision variables based on the variables and their types recorded in var_names
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
        Define cost function, default is 0
        """
        self.cost = 0

    def get_constraints(self):
        """
        Define constraint list, default is empty
        """
        pass

    def add_on_off_constraints(self):
        """
        Add start-up/shutdown constraints
        """
        self.constraints += [
            self.variables['v'][1:] - self.variables['v'][:-1] == self.variables['u_on'][1:] - self.variables['u_off'][
                                                                                               1:],
            self.variables['u_on'] + self.variables['u_off'] <= 1
        ]
        for h in range(self.nT):
            tau_on = min(self.nT - h, self.config['tau_on'])  # Minimum uptime
            tau_off = min(self.nT - h, self.config['tau_off'])  # Minimum downtime
            self.constraints += [
                cp.sum([self.variables['v'][h + t] for t in range(0, tau_on)]) >= self.variables['u_on'][h] * tau_on,
                cp.sum([(1 - self.variables['v'][h + t]) for t in range(0, tau_off)]) >= self.variables['u_off'][
                    h] * tau_off]


class PowerLoad(Device):
    """
    Powerload
    """

    def __init__(self, name: str, nT: int, dT: float, config: dict):
        super().__init__(name, nT, dT, config=config, vars_={})

    def get_data(self):
        """
        Read load data and adjust according to parameters
        """
        self.config['P_load'] = pd.read_csv(self.config['path'])[str(self.config['index_col'])].values[::self.delta_T] * \
                                self.config[
                                    'magnification factor']

    def get_constraints(self):
        """
        The electric output of the load is negative, and the thermal output is 0
        """
        self.variables['P'] = - self.config['P_load']
        self.variables['Q'] = 0
        self.variables['H'] = 0


class QLoad(Device):
    """
    Reactive power load
    """

    def __init__(self, name: str, nT: int, dT: float, config: dict):
        super().__init__(name, nT, dT, config=config, vars_={})

    def get_data(self):
        """
        Read load data and adjust according to parameters
        """
        self.config['Q_load'] = pd.read_csv(self.config['path'])[str(self.config['index_col'])].values[::self.delta_T] * \
                                self.config[
                                    'magnification factor']

    def get_constraints(self):
        self.variables['P'] = 0
        self.variables['Q'] = - self.config['Q_load']
        self.variables['H'] = 0


class HeatLoad(Device):
    """
    heat load
    """

    def __init__(self, name: str, nT: int, dT: float, config: dict):
        super().__init__(name, nT, dT, config=config, vars_={})

    def get_data(self):
        """
        Read load data and adjust according to parameters
        """
        self.config['H_load'] = pd.read_csv(self.config['path'])[str(self.config['index_col'])].values[::self.delta_T] * \
                                self.config[
                                    'magnification factor']

    def get_constraints(self):
        """
        The heat output of the load is negative
        """
        self.variables['P'] = 0
        self.variables['Q'] = 0
        self.variables['H'] = - self.config['H_load']


class PV(Device):
    """
    Photovoltaic
    """

    def __init__(self, name: str, nT: int, dT: float, config: dict):
        super().__init__(name, nT, dT, config=config, vars_={'P': 'float', 'Q': 'float'})

    def get_data(self):
        """
        Read PV data and adjust according to parameters
        """
        self.config['P_max'] = pd.read_csv(self.config['path'])[str(self.config['index_col'])].values[::self.delta_T] * \
                               self.config[
                                   'magnification factor']

    def get_constraints(self):
        """
        PV maximum output constraint, thermal output is 0
        """
        self.variables['H'] = 0
        self.constraints += [
            self.variables['P'] >= 0,
            self.variables['P'] <= self.config['P_max'],
            self.variables['Q'] <= self.variables['P'] / 0.9 * (1 - 0.9 ** 2) ** 0.5,
            self.variables['Q'] >= -self.variables['P'] / 0.9 * (1 - 0.9 ** 2) ** 0.5]


class WP(Device):
    """
    Wind power
    """

    def __init__(self, name: str, nT: int, dT: float, config: dict):
        super().__init__(name, nT, dT, config=config, vars_={'P': 'float', 'Q': 'float'})

    def get_data(self):
        """
        Read wind power data and adjust according to parameters
        """
        self.config['P_max'] = pd.read_csv(self.config['path'])[str(self.config['index_col'])].values[::self.delta_T] * \
                               self.config['magnification factor']

    def get_constraints(self):
        """
        Wind power maximum output constraint, thermal output is 0
        """
        self.variables['H'] = 0
        self.constraints += [
            self.variables['P'] >= 0,
            self.variables['P'] <= self.config['P_max'],
            self.variables['Q'] >= -self.variables['P'] / 0.9 * (1 - 0.9 ** 2) ** 0.5,
            self.variables['Q'] <= self.variables['P'] / 0.9 * (1 - 0.9 ** 2) ** 0.5]


class CHP(Device):
    """
    Combined heat and power unit
    """

    def __init__(self, name: str, nT: int, dT: float, config: dict):
        """
        Device parameters: electric output, thermal output
        """
        super().__init__(name, nT, dT, config=config, vars_={
            'P': 'float',
            'H': 'float',
            'Q': 'float'})

    def get_constraints(self):
        """
        CHP electric output constraint, thermal output constraint, and start-up/shutdown constraints if applicable
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
        CHP operating cost function
        """
        self.cost = cp.sum(self.config['gas_p'] * self.variables['P'] * (1 + self.config['alpha'])) * self.dT



class Battery(Device):
    """
    Battery
    """

    def __init__(self, name: str, nT: int, dT: float, config: dict):
        """
        Device parameters: SOC, charging power, discharging power
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
        Battery SOC constraint, charging power constraint, discharging power constraint, and start-up/shutdown constraints if applicable
        """
        self.variables['P'] = self.variables['P_discharge'] - self.variables['P_charge']  # Net power generation
        self.variables['Q'] = self.variables['Q_discharge'] - self.variables['Q_charge']
        self.variables['H'] = 0
        self.config['P_max'] = self.config['E_max'] / 3

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
            self.variables['E'][-1] == self.config['E_max'] * 0.3,
            self.variables['E'] - cp.hstack([self.config['E_max'] * 0.3, self.variables['E'][:-1]]) * (
                    1 - self.config['r_loss'] * self.dT) == self.variables['P_charge'] * self.config['eta_charge'] -
            self.variables['P_discharge'] / self.config['eta_discharge']
        ]


class HeatStorage(Device):
    """
    Heat storage
    """

    def __init__(self, name: str, nT: int, dT: float, config: dict):
        """
        Device parameters: SOC, thermal storage power, heat release power
        """
        super().__init__(name, nT, dT, config=config, vars_={
            'E': 'float',
            'H_rel': 'float',
            'H_sto': 'float'})

    def get_constraints(self):
        """
        Thermal storage constraints: storage power, heat release power, SOC
        """
        self.variables['P'] = 0
        self.variables['Q'] = 0
        self.config['H_max'] = self.config['E_max'] / 3
        self.variables['H'] = self.variables['H_rel'] - self.variables['H_sto']  # Net heat generation
        self.constraints += [
            self.variables['H_rel'] <= self.config['H_max'],
            self.variables['H_rel'] >= 0,
            self.variables['H_sto'] >= 0,
            self.variables['H_sto'] <= self.config['H_max'],
            self.variables['E'] >= 0,
            self.variables['E'] <= self.config['E_max'],
            self.variables['E'][-1] == self.config['E_max'] * 0.3,
            self.variables['E'] - cp.hstack([self.config['E_max'] * 0.3, self.variables['E'][:-1]]) * (
                    1 - self.config['r_loss'] * self.dT) == self.variables['H_sto'] - self.variables['H_rel']
        ]


class ElectricHeatPump(Device):
    """
    ElectricHeatPump
    """

    def __init__(self, name: str, nT: int, dT: float, config: dict):
        """
        Device parameters: electric output, thermal output
        """
        super().__init__(name, nT, dT, config=config, vars_={
            'P': 'float',
            'H': 'float',
            'Q': 'float'})

    def get_constraints(self):
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


class ElectricHeater(Device):
    """
    ElectricHeater
    """

    def __init__(self, name: str, nT: int, dT: float, config: dict):
        super().__init__(name, nT, dT, config=config, vars_={
            'P': 'float',
            'H': 'float'})

    def get_constraints(self):
        self.variables['Q'] = 0
        self.constraints += [
            self.variables['H'] == - self.variables['P'] * self.config['eta']
        ]
        if 'v' in self.variables.keys():
            self.constraints += [self.variables['P'] >= - self.variables['v'] * self.config['P_max'],
                                 self.variables['P'] <= - self.variables['v'] * self.config['P_min']]
        else:
            self.constraints += [self.variables['P'] >= - self.config['P_max'],
                                 self.variables['P'] <= - self.config['P_min']]
        if self.on_off_flag:
            self.add_on_off_constraints()


class GridInterface(Device):
    """
    GridInterface
    """

    def __init__(self, name: str, nT: int, dT: float, config: dict):
        super().__init__(name, nT, dT, config=config, vars_={
            'P_buy': 'float',
            'P_sell': 'float',
            'Q_buy': 'float',
            'Q_sell': 'float',
            'Q': 'float'})

    def get_data(self):
        """
        Read purchase and sale electricity price curves
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
        Constraints on power purchase and sale between the device and the grid
        """
        self.variables['P'] = self.variables['P_buy'] - self.variables['P_sell']
        self.variables['Q'] = self.variables['Q_buy'] - self.variables['Q_sell']
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
        Cost of power purchase and sale between the device and the grid
        """
        self.cost = cp.sum(
            cp.multiply(self.config['price_buy'], self.variables['P_buy']) - cp.multiply(self.config['price_sell'],
                                                                                         self.variables[
                                                                                             'P_sell'])) * self.dT
