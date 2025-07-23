import copy
import json
import tenseal as ts
import numpy as np
from device_lib import *

nT = 24  # number of time steps in a day
n_DES = 7
dT = 24 / nT  # time interval in hours
delta_T = int(96 / nT)

# Device dictionary: maps short names to device classes for quick lookup
ds = {
    "PV": PV,
    "WP": WP,
    "BT": Battery,
    "CHP": CHP,
    "HS": HeatStorage,
    "GI": GridInterface,
    "EH": ElectricHeater,
    "EHP": ElectricHeatPump,
    "P_load": PowerLoad,
    "H_load": HeatLoad,
    "Q_load": QLoad
}


class Entity:  # Base class for all entities
    """
    DES base class: provides fundamental attributes and methods for all entities, plus encryption/decryption utilities
    """

    def __init__(self, entity_type: str, entity_id: int, entity_config: str, n_parent: int = 0, n_child: int = 0,
                 st=False):
        self.type = entity_type
        self.id = entity_id
        self.st = st
        self.price_buy = pd.read_csv('data/6-bus/price.csv').values[::delta_T, 0]
        self.pub_keys_A = dict()  # Encryption key list, own public key
        self.pub_keys_B = dict()  # Encryption key list, neighbor's public key
        self.sec_keys_A = dict()  # Private key list, only one's own private key
        self.enc_external_variables_A = dict()  # Encrypted external variables
        self.enc_external_variables_B = dict()
        self.enc_external_variables_A_add = dict()  # The result of adding ciphertexts
        with open(entity_config, "r") as f:
            self.config = json.load(f)["devices"]
        self.n_parent = n_parent
        self.n_child = n_child
        self.generate_keys()
        self.model = None
        self.rho = 1e-1
        self.objective = None
        self.cost = None
        self.penalty = None
        self.pri_error = 1e6
        self.dual_error = 1e6
        self.line_impedance = dict()  # Line impedance
        self.alpha = dict()  # Weight alpha
        self.P_ref = 1000  # kW
        self.R_ref = 100  # ohm
        self.devices = dict()  # Dictionary of all devices in the system
        self.variables = dict()  # Dictionary of all variables in the system, such as P_parent, P_child, etc
        self.constraints = list()  # A list of all constraints in the system, including internal constraints of all devices and overall electric thermal balance constraints within DES
        self.external_variables = dict()  # Coupled variables
        self.enc_external_variables = dict()  # Encrypted coupled variables
        self.lambdas = dict()
        self.results = dict()
        self.results_last = dict()
        self.results_from_masked_ones = dict()
        self.get_variables()
        self.add_devices()

    def __str__(self):
        return f"({self.type})-({self.id})"

    def __repr__(self):
        return f"({self.type})-({self.id})"

    def generate_keys(self):
        """
        Generate encryption key
        """
        for i in range(self.n_parent):
            # Generate key pairs: A = locally generated private/public pair, B = neighbor's public key
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[60, 40, 40, 60])
            context.generate_galois_keys()
            context.global_scale = 2 ** 40
            sec_A = context.secret_key()
            context.make_context_public()
            pub_A = context
            self.pub_keys_A.update({'parent': pub_A})
            self.sec_keys_A.update({'parent': sec_A})

        for i in range(self.n_child):
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[60, 40, 40, 60])
            context.generate_galois_keys()
            context.global_scale = 2 ** 40
            sec_A = context.secret_key()
            context.make_context_public()
            pub_A = context
            self.pub_keys_A.update({f'child{i}': pub_A})
            self.sec_keys_A.update({f'child{i}': sec_A})

    def get_variables(self):
        """
        Initialize all variables
        """

        self.variables['psell'] = cp.Variable(nT)
        self.variables['pbuy'] = cp.Variable(nT)
        if self.n_parent == 1:
            self.lambdas["P_parent"] = self.price_buy * dT
            self.lambdas["Q_parent"] = np.zeros(nT)
            self.lambdas["parent_voltage"] = np.zeros(nT)
            self.external_variables["P_parent"] = np.zeros(nT)
            self.external_variables['Q_parent'] = np.zeros(nT)
            self.external_variables['parent_voltage'] = np.zeros(nT)
            self.enc_external_variables['P_parent'] = np.zeros(3 * nT)
            self.variables["P_parent"] = cp.Variable(nT)
            self.variables["Q_parent"] = cp.Variable(nT)
            self.variables['voltage'] = cp.Variable(nT)
            self.variables['parent_voltage'] = cp.Variable(nT)
            self.results["P_parent"] = np.zeros(nT)
            self.results["Q_parent"] = np.zeros(nT)
            self.results['parent_voltage'] = np.zeros(nT)
        else:
            self.variables['voltage'] = np.ones(nT)

        if self.n_child > 0:
            self.lambdas["P_child"] = - np.tile(self.price_buy.reshape(-1, 1), (1, self.n_child)) * dT
            self.lambdas["Q_child"] = np.zeros((nT, self.n_child))
            self.lambdas["child_voltage"] = np.zeros((nT, self.n_child))
            self.external_variables["P_child"] = np.zeros((nT, self.n_child))
            self.external_variables['Q_child'] = np.zeros((nT, self.n_child))
            self.external_variables['child_voltage'] = np.zeros((nT, self.n_child))
            self.enc_external_variables['P_child'] = np.zeros((3 * nT, self.n_child))
            self.variables["P_child"] = cp.Variable((nT, self.n_child))
            self.variables["Q_child"] = cp.Variable((nT, self.n_child))
            self.variables['child_voltage'] = cp.Variable((nT, self.n_child))
            self.results["P_child"] = np.zeros((nT, self.n_child))
            self.results["Q_child"] = np.zeros((nT, self.n_child))
            self.results['child_voltage'] = np.zeros((nT, self.n_child))
        self.results['voltage'] = np.ones(nT)

    def add_devices(self):
        """
        initialize constraints for all devices.
        """
        for device, config in self.config.items():
            self.devices[device] = ds[device](f"({self.type})-({self.id})-({device})", nT, dT, config)

    def get_constraints(self):
        """
        initialize all constraints
        """
        self.constraints = list()
        # Internal Constraints
        for device in self.devices.values():
            self.constraints.extend(device.constraints)
        # Energy Conservation of devices
        self.variables["P"] = cp.sum([device.variables["P"] for device in self.devices.values()])
        self.variables["H"] = cp.sum([device.variables["H"] for device in self.devices.values()])
        self.variables["Q"] = cp.sum([device.variables["Q"] for device in self.devices.values()])
        # Energy Conservation of DES and power flow
        P_net = 0
        Q_net = 0
        if self.n_parent == 1:
            P_net += self.variables["P_parent"]
            Q_net += self.variables["Q_parent"]
            self.constraints += [self.variables["P_parent"] >= -1e5, self.variables["P_parent"] <= 1e5]
            self.constraints += [self.variables['Q_parent'] >= -1e5, self.variables['Q_parent'] <= 1e5]
            self.constraints += [self.variables["voltage"] >= 0.95, self.variables["voltage"] <= 1.05]
            self.constraints += [self.variables['parent_voltage'] >= 0.95, self.variables['parent_voltage'] <= 1.05]
            self.constraints += [
                self.variables["voltage"] == self.variables['parent_voltage'] - self.variables[
                    "P_parent"] / self.P_ref *
                self.line_impedance['parent']['R'] / self.R_ref - self.variables["Q_parent"] / self.P_ref *
                self.line_impedance['parent']['X'] / self.R_ref]
        self.constraints += [self.variables['voltage'] == self.variables['child_voltage'][:, i] for i in
                             range(self.n_child)]
        if self.n_child > 0:
            P_net += - cp.sum(self.variables["P_child"], axis=1)
            Q_net += - cp.sum(self.variables["Q_child"], axis=1)
            self.constraints += [self.variables["P_child"] >= -1e5, self.variables["P_child"] <= 1e5]
            self.constraints += [self.variables['Q_child'] >= -1e5, self.variables['Q_child'] <= 1e5]
        # Energy Conservation
        self.constraints += [self.variables["P"] + P_net == 0]  # 电力有功平衡约束
        self.constraints += [self.variables["Q"] + Q_net == 0]  # 无功平衡约束
        self.constraints += [self.variables["H"] == 0]  # 热力平衡约束
        # Costs
        if self.n_parent == 1:
            self.constraints += [self.variables["psell"] - self.variables["pbuy"] == self.variables["P"]]  # 电力平衡约束
            self.constraints += [self.variables["psell"] >= 0, self.variables["psell"] <= 1e5]
            self.constraints += [self.variables["pbuy"] >= 0, self.variables["pbuy"] <= 1e5]
            # taxes
            self.tax = cp.sum(self.variables['psell'] * 0.244343 * dT)
        # Costs
        if self.n_parent == 1:
            self.cost = cp.sum([device.cost for device in self.devices.values()]) + self.tax
        else:
            self.cost = cp.sum([device.cost for device in self.devices.values()])

    def get_objective(self):
        """
        Objective Function
        """
        # lagrangian penalties
        self.penalty = 0
        if self.n_parent == 1:
            P_parent_diff = self.variables["P_parent"] - self.external_variables["P_parent"]
            self.penalty += cp.sum(
                cp.multiply(self.lambdas["P_parent"], P_parent_diff)) + 0.5 * self.rho * cp.sum_squares(
                P_parent_diff)
            Q_parent_diff = self.variables["Q_parent"] - self.external_variables["Q_parent"]
            self.penalty += cp.sum(
                cp.multiply(self.lambdas["Q_parent"], Q_parent_diff)) + 0.5 * self.rho * cp.sum_squares(
                Q_parent_diff)
            parent_voltage_diff = self.variables['parent_voltage'] - self.external_variables['parent_voltage']
            self.penalty += cp.sum(
                cp.multiply(self.lambdas["parent_voltage"], parent_voltage_diff)) + 0.5 * self.rho_u * cp.sum_squares(
                parent_voltage_diff)
        if self.n_child > 0:  # child nodes
            P_child_diff = self.variables["P_child"] - self.external_variables["P_child"]
            Q_child_diff = self.variables["Q_child"] - self.external_variables["Q_child"]
            child_voltage_diff = self.variables['child_voltage'] - self.external_variables['child_voltage']
            for i in range(self.n_child):
                self.penalty += cp.sum(
                    cp.multiply(self.lambdas["P_child"][:, i], P_child_diff[:, i])) + 0.5 * self.rho * cp.sum_squares(
                    P_child_diff[:, i])
                self.penalty += cp.sum(
                    cp.multiply(self.lambdas["Q_child"][:, i], Q_child_diff[:, i])) + 0.5 * self.rho * cp.sum_squares(
                    Q_child_diff[:, i])
                self.penalty += cp.sum(
                    cp.multiply(self.lambdas["child_voltage"][:, i],
                                child_voltage_diff[:, i])) + 0.5 * self.rho_u * cp.sum_squares(
                    child_voltage_diff[:, i])
        # overall objective
        self.objective = self.cost + self.penalty

    def build_model(self):
        """
        Build original Model and RMME Model
        """
        self.model = cp.Problem(cp.Minimize(self.objective), self.constraints)
        all_vars_map = 0.0
        coupling_vars_map = 0.0
        if self.st:
            data, chain, inverse_data = self.model.get_problem_data(cp.OSQP)
            # min (1/2)x'Hx + c'x + a s.t. Ax = b, Gx <= h
            H = data['P'].toarray()
            c = data['q']
            A = data['A'].toarray()
            b = data['b']
            G = data['F'].toarray()
            h = data['G']

            # x dimension
            n = A.shape[0]
            q, r = np.linalg.qr(A.T, mode='complete')
            C = q[:, 0:n]
            N = q[:, n:]
            x0 = C @ np.linalg.inv(r[0:A.shape[0], 0:A.shape[0]].T) @ b
            from scipy import stats
            R = np.eye(A.shape[1] - A.shape[0]) @ stats.ortho_group.rvs(A.shape[1] - A.shape[0])
            H_pre = R.T @ N.T @ H @ N @ R
            G_pre = G @ N @ R
            h_pre = h - G @ x0
            c_pre = ((c + x0.T @ H) @ N @ R)
            a_pre = (1 / 2 * x0 @ H @ x0 + c.T @ x0)
            # RMME decision variables
            self.y = cp.Variable((N.shape[1]))
            self.goal_mask = 0.5 * cp.quad_form(self.y, H_pre) + c_pre.T @ self.y + a_pre
            constraints_mask = [G_pre @ self.y <= h_pre]
            self.model_mask = cp.Problem(cp.Minimize(self.goal_mask), constraints_mask)
            # recovered variables
            self.x_from_y = N @ R @ self.y + x0
            # map between coupling variables and RMME vector
            Cij, all_vars_map, coupling_vars_map, total_vars_dim = self.get_coupling_variable_selector(data, chain,
                                                                                                       inverse_data)
            self.NR = N @ R
            self.x0 = x0
            self.Cij = Cij
        return all_vars_map, coupling_vars_map

    def get_coupling_variable_selector(self, data, chain, inverse_data):
        """
        generate Cij for xc=(P_parent, Q_parent, P_child, Q_child)
        such that xc = Cij @ x,

        Return:
            Cij (np.ndarray):
            all_vars_map (dict): a dict mapping name to x
            coupling_vars_map (dict): a dict mapping name to xc
            total_vars_dim (int):
        """
        all_vars_list = self.model.variables()

        all_vars_map = {}
        current_pos = 0
        for id in inverse_data[1].id2var.keys():
            var = inverse_data[2].id2var[id]
            var_size = var.size
            all_vars_map[var.name()] = {'var': var, 'start': inverse_data[2].id_map[id][0],
                                        'end': inverse_data[2].id_map[id][0] + var_size}

        total_vars_dim = data['A'].shape[1]
        # print(f"dimensions of original model (x): {total_vars_dim}")

        coupling_vars_to_find = []
        if self.n_parent == 1:
            coupling_vars_to_find.extend(["P_parent", "Q_parent", "parent_voltage"])
        if self.n_child > 0:
            coupling_vars_to_find.extend(["P_child", "Q_child", "child_voltage"])

        target_vars = {name: self.variables[name] for name in coupling_vars_to_find}

        total_coupling_dim = sum(v.size for v in target_vars.values())
        # print(f"dimension of coupling variables (xc): {total_coupling_dim}")

        Cij = np.zeros((total_coupling_dim, total_vars_dim))

        coupling_vars_map = {}
        current_coupling_pos = 0

        for var_key in sorted(target_vars.keys()):
            var = target_vars[var_key]
            var_name = var.name()

            if var_name not in all_vars_map:
                print(f"Warning：Variable '{var_name}' not found")
                continue

            # locate the variable in x
            start_idx_in_x = all_vars_map[var_name]['start']
            end_idx_in_x = all_vars_map[var_name]['end']

            # locate the variable in xc
            var_size = var.size
            start_idx_in_xc = current_coupling_pos
            end_idx_in_xc = current_coupling_pos + var_size

            coupling_vars_map[var_key] = {'start': start_idx_in_xc, 'end': end_idx_in_xc}

            Cij[start_idx_in_xc:end_idx_in_xc, start_idx_in_x:end_idx_in_x] = np.eye(var_size)

            current_coupling_pos = end_idx_in_xc

        return Cij, all_vars_map, coupling_vars_map, total_vars_dim

    def get_results(self):
        """
        Retrieve optimization results
        """
        if self.st:
            self.results["P_parent"] = self.results_from_masked_ones['P_parent'] if self.n_parent == 1 else 0
            self.results["P_child"] = self.results_from_masked_ones['P_child'] if self.n_child > 0 else 0
            self.results["Q_parent"] = self.results_from_masked_ones['Q_parent'] if self.n_parent == 1 else 0
            self.results["Q_child"] = self.results_from_masked_ones['Q_child'] if self.n_child > 0 else 0
            # self.results['voltage'] = self.variables['voltage'].value if self.n_parent == 1 else np.ones(nT)
            self.results['parent_voltage'] = self.results_from_masked_ones[
                'parent_voltage'] if self.n_parent == 1 else np.zeros(nT)
            self.results['child_voltage'] = self.results_from_masked_ones[
                'child_voltage'] if self.n_child > 0 else np.zeros((nT, self.n_child))
            self.results["obj"] = self.results_from_masked_ones['obj'] if isinstance(self.objective, cp.Expression) else 0
        else:
            self.results["P_parent"] = self.variables["P_parent"].value if self.n_parent == 1 else 0
            self.results["P_child"] = self.variables["P_child"].value if self.n_child > 0 else 0
            self.results["Q_parent"] = self.variables["Q_parent"].value if self.n_parent == 1 else 0
            self.results["Q_child"] = self.variables["Q_child"].value if self.n_child > 0 else 0
            # self.results['voltage'] = self.variables['voltage'].value if self.n_parent == 1 else np.ones(nT)
            self.results['parent_voltage'] = self.variables[
                'parent_voltage'].value if self.n_parent == 1 else np.zeros(nT)
            self.results['child_voltage'] = self.variables[
                'child_voltage'].value if self.n_child > 0 else np.zeros((nT, self.n_child))
            self.results["obj"] = self.objective.value if isinstance(self.objective, cp.Expression) else 0
        self.results['P_parent_diff'] = self.results['P_parent'] - self.external_variables[
            "P_parent"] if self.n_parent == 1 else 0
        self.results['Q_parent_diff'] = self.results['Q_parent'] - self.external_variables[
            "Q_parent"] if self.n_parent == 1 else 0
        self.results['P_child_diff'] = self.results['P_child'] - self.external_variables[
            "P_child"] if self.n_child > 0 else 0
        self.results['Q_child_diff'] = self.results['Q_child'] - self.external_variables[
            "Q_child"] if self.n_child > 0 else 0
        self.results["alpha_parent"] = 0 if self.n_parent == 1 else None
        self.results["alpha_child"] = np.zeros(self.n_child) if self.n_child > 0 else None
        return self.results

    def get_results_from_masked_ones(self, all_vars_map, coupling_vars_map):
        """
        获取优化结果
        """
        target_vars = ['voltage'] if self.n_parent == 1 else []
        self.results_from_masked_ones['obj'] = self.goal_mask.value
        if self.n_parent == 1:
            target_vars.extend(["P_parent", "Q_parent", "parent_voltage"])
        if self.n_child > 0:
            target_vars.extend(["P_child", "Q_child", "child_voltage"])
        for calling_name in target_vars:
            internal_name = self.variables[calling_name].name()
            shape = self.variables[calling_name].shape
            start_position = all_vars_map[internal_name]['start']
            end_position = all_vars_map[internal_name]['end']
            self.results_from_masked_ones[calling_name] = self.x_from_y.value[start_position:end_position].reshape(
                shape, order='F')
        return self.results_from_masked_ones

    def update_model(self, i, rho_u):
        """
        Update the optimization model, solve, and record results
        """
        # Store results from the previous iteration
        self.results_last = copy.deepcopy(self.results)
        self.rho = i
        self.rho_u = rho_u
        # Record the primal residual
        self.pri_error = 0.0
        if self.n_parent == 1:
            P_parent_diff = self.results["P_parent"] - self.external_variables["P_parent"]
            Q_parent_diff = self.results["Q_parent"] - self.external_variables["Q_parent"]
            parent_voltage_diff = self.results['parent_voltage'] - self.external_variables['parent_voltage']
            self.lambdas["P_parent"] += self.rho * P_parent_diff / self.alpha['parent'] * 0.5
            self.lambdas["Q_parent"] += self.rho * Q_parent_diff / self.alpha['parent'] * 0.5
            self.lambdas["parent_voltage"] += self.rho_u * parent_voltage_diff / self.alpha['parent'] * 0.5
            self.pri_error = max(self.pri_error, np.linalg.norm(P_parent_diff))
            self.pri_error = max(self.pri_error, np.linalg.norm(parent_voltage_diff))
        if self.n_child > 0:
            P_child_diff = self.results["P_child"] - self.external_variables["P_child"]
            Q_child_diff = self.results["Q_child"] - self.external_variables["Q_child"]
            child_voltage_diff = self.results['child_voltage'] - self.external_variables['child_voltage']
            self.lambdas["P_child"] += self.rho * P_child_diff / (
                        1 - np.array([self.alpha[f"child{i}"] for i in range(self.n_child)])) * 0.5
            self.lambdas["Q_child"] += self.rho * Q_child_diff / (
                        1 - np.array([self.alpha[f"child{i}"] for i in range(self.n_child)])) * 0.5
            self.lambdas["child_voltage"] += self.rho_u * child_voltage_diff / (
                        1 - np.array([self.alpha[f"child{i}"] for i in range(self.n_child)])) * 0.5
            self.pri_error = max(self.pri_error, np.linalg.norm(P_child_diff.flatten()))
            self.pri_error = max(self.pri_error, np.linalg.norm(child_voltage_diff.flatten()))
        # update the model
        self.get_objective()
        all_vars_map, coupling_vars_map = self.build_model()
        # self.model.solve(solver=cp.OSQP)
        self.model_mask.solve(solver=cp.OSQP) if self.st else self.model.solve(solver=cp.OSQP)
        #
        self.get_results_from_masked_ones(all_vars_map, coupling_vars_map) if self.st else self.get_results()
        self.get_results()
        self.encrypt_external_variables()
        # record errors
        self.dual_error = 0.0
        if self.n_parent == 1:
            P_parent_diff = self.results["P_parent"] - self.results_last["P_parent"]
            Q_parent_diff = self.results["Q_parent"] - self.results_last["Q_parent"]
            parent_voltage_diff = self.results['parent_voltage'] - self.results_last['parent_voltage']
            self.dual_error = max(self.dual_error, np.linalg.norm(P_parent_diff))
            self.dual_error = max(self.dual_error, np.linalg.norm(parent_voltage_diff))
        if self.n_child > 0:
            P_child_diff = self.results["P_child"] - self.results_last["P_child"]
            Q_child_diff = self.results["Q_child"] - self.results_last["Q_child"]
            child_voltage_diff = self.results['child_voltage'] - self.results_last['child_voltage']
            self.dual_error = max(self.dual_error, np.linalg.norm(P_child_diff.flatten()))
            self.dual_error = max(self.dual_error, np.linalg.norm(child_voltage_diff.flatten()))

    def encrypt_external_variables(self):

        """
        Encrypt coupling variables using encrypt_by_key_parent
        """
        if self.n_parent == 1:
            a = np.zeros(3 * nT)
            a[:nT] = (self.results["P_parent"])
            a[nT:2 * nT] = (self.results["Q_parent"])
            a[2 * nT:3 * nT] = (self.results["parent_voltage"])
            self.enc_external_variables_A["P_parent"] = self.encrypt_by_key_parent(a,
                                                                                   self.pub_keys_A) if self.st else a
            self.enc_external_variables_B["P_parent"] = self.encrypt_by_key_parent(a,
                                                                                   self.pub_keys_B) if self.st else a
        if self.n_child > 0:
            b = np.zeros((3 * nT, self.n_child))
            for i in range(self.n_child):
                b[:nT, i] = (self.results["P_child"][:, i])
                b[nT:2 * nT, i] = (self.results["Q_child"][:, i])
                b[2 * nT:3 * nT, i] = (self.results["child_voltage"][:, i])

            self.enc_external_variables_A["P_child"] = self.encrypt_by_key_child(b, self.pub_keys_A)
            self.enc_external_variables_B["P_child"] = self.encrypt_by_key_child(b, self.pub_keys_B)

    def decrypt_external_variables(self):

        """
        Decrypt auxiliary variables
        :return:
        """
        if self.n_parent == 1:
            external_varibles = self.enc_external_variables_A_add["P_parent"].decrypt(
                self.sec_keys_A['parent']) if self.st else self.enc_external_variables_A_add["P_parent"]
            self.external_variables["P_parent"] = np.array(external_varibles[:nT])
            self.external_variables["Q_parent"] = np.array(external_varibles[nT:2 * nT])
            self.external_variables["parent_voltage"] = np.array(external_varibles[2 * nT:3 * nT])

        if self.n_child > 0:
            for i in range(self.n_child):
                external_varibles = self.enc_external_variables_A_add["P_child"][i].decrypt(
                    self.sec_keys_A[f'child{i}']) if self.st else self.enc_external_variables_A_add["P_child"][
                    i]
                self.external_variables["P_child"][:, i] = np.array(external_varibles[:nT])
                self.external_variables["Q_child"][:, i] = np.array(external_varibles[nT:2 * nT])
                self.external_variables["child_voltage"][:, i] = np.array(external_varibles[2 * nT:3 * nT])

    def encrypt_by_key_parent(self, value, keys):
        """
        encrypt
        """

        encrypted_value_parent = ts.ckks_vector(keys['parent'], value)
        return encrypted_value_parent

    def encrypt_by_key_child(self, value, keys):
        """
       encrypt
       """
        encrypted_value = [0] * self.n_child
        for i in range(self.n_child):
            encrypted_value[i] = ts.ckks_vector(keys[f'child{i}'], value[:, i]) if self.st else value[:, i]
        return encrypted_value


class DistributedEnergySystem(Entity):  # DES class
    """
    Distributed Energy System class.
    """

    def __init__(self, entity_id: int, entity_config: str, n_parent: int = 0, n_child: int = 0):
        super().__init__("DES", entity_id, entity_config, n_parent, n_child)
