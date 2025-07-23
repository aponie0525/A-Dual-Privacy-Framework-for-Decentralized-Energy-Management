from dso_lib import *
from des_lib import *
import time
import multiprocessing
from functools import partial


class SimulationSystem:
    def __init__(self, network_config_path_, des_config_path_):
        self.log_info = dict()
        self.cost_dict = dict()
        # Voltage record
        self.voltage_dict = dict()
        self.electrical_cost_dict = dict()
        self.electrical_cost_dict_u = dict()
        self.network_config_path = network_config_path_
        self.des_config_path = des_config_path_
        # Instantiate a distribution network information management system
        self.dso = DistributionSystemOperator(config_path=network_config_path_)
        self.des_ = {i:
                         DistributedEnergySystem(entity_id=node, entity_config=des_config_path_[i],
                                                 n_parent=self.dso.n_parent[node],
                                                 n_child=self.dso.n_child[node]) for i, node in
                     enumerate(self.dso.graph.nodes)}

    def exchange_info_in_dso(self, i):
        """
        Collect ciphertexts from various DES in DSO, and exchange and update this information
        :return:
        """
        self.collect_results_from_des()  # Collect the results of each DES
        self.dso.update_edge_from_nodes(i)  # Update the z information of the edge
        self.dso.update_nodes_from_edges()  # Update the p-parent and p-child information of the node
        self.load_results_from_dso()  # Load the results of DSO into DES
        self.decrypt_results_in_des()  # Each DES completes decryption

    def opt_cost_in_des(self, rho, rho_u):
        """
        Optimize local costs in each DES
        :return:
        """
        for des in self.des_.values():
            des.update_model(rho, rho_u)

    def opt_cost_in_des_parallel(self, rho):
        """
        Optimize local costs in various DES, parallel versions
        :return:
        """
        for des_results in _map(partial(run_instance, des_list=self.des_, rho=rho), [0, 30]):
            for i, result in des_results.items():
                self.des_[i].results = result['results']
                self.des_[i].dual_error = result['dual_error']
                self.des_[i].pri_error = result['pri_error']
                self.des_[i].lambdas = result['lambda']

    def collect_results_from_des(self):
        for i, node in zip(range(len(self.des_)), self.dso.graph.nodes):
            self.dso.graph.nodes[node]['des'].update_x_from_des_A(copy.deepcopy(self.des_[i].enc_external_variables_A))
            self.dso.graph.nodes[node]['des'].update_x_from_des_B(copy.deepcopy(self.des_[i].enc_external_variables_B))

    def load_results_from_dso(self):
        for i, node in zip(range(len(self.des_)), self.dso.graph.nodes):
            self.des_[i].enc_external_variables_A_add = copy.deepcopy(
                self.dso.graph.nodes[node]['des'].results_A)

    def decrypt_results_in_des(self):
        """
        Each edge device completes decryption
        :return:
        """
        for i, node in zip(range(len(self.des_)), self.dso.graph.nodes):
            self.des_[i].decrypt_external_variables()

    def exchange_pub_keys_through_dso(self):
        """
        DSO collects the public keys of each DES and distributes them to each DES based on the power grid topology
        :return:
        """
        self.collect_pub_keys_from_des()  # Collect the public keys of each DES
        self.dso.distribute_public_keys()  # DSO distributes public key B to each DES
        self.load_pub_keys_B_from_dso()  # DES loads the neighbor's public key (i.e. public key B) from the DSO information container

    def collect_pub_keys_from_des(self):
        """
        Collect the public key A of each DES
        :return:
        """
        for i, node in zip(range(len(self.des_)), self.dso.graph.nodes):
            self.dso.graph.nodes[node]['des'].add_keys_A(copy.deepcopy(self.des_[i].pub_keys_A))

    def load_pub_keys_B_from_dso(self):
        """
        DES loads the neighbor's public key(i.e. public key B) from the DES container of DSO
        """
        for i, node in zip(range(len(self.des_)), self.dso.graph.nodes):
            self.des_[i].pub_keys_B = copy.deepcopy(self.dso.graph.nodes[node]['des'].pub_keys_B)

    def distribute_line_impedance(self):
        """
        DSO distributes line impedance to each DES
        :return:
        """
        self.dso.distribute_line_impedance_from_edges()
        self.dso.distribute_alpha_from_nodes()
        for i, node in zip(range(len(self.des_)), self.dso.graph.nodes):
            self.des_[i].line_impedance = copy.deepcopy(self.dso.graph.nodes[node]['des'].line_impedance)
            self.des_[i].get_constraints()
            self.des_[i].alpha = copy.deepcopy(self.dso.graph.nodes[node]['des'].alpha)

    def run(self):
        """
        Run the entire simulation system
        :return:
        """
        self.exchange_pub_keys_through_dso()  # Exchange public keys through DSO
        self.distribute_line_impedance()
        i = 0
        tick = time.time()
        self.log_info = {
            'iteration': list(),
            'time': list(),
            'rho': list(),
            'dual_error': list(),
            'pri_error': list(),
            'obj': list(),
            'cost': list(),
            'penalty': list()
        }
        self.cost_dict = {'iteration': list()}
        self.cost_dict.update({f"des{i}": list() for i in range(n_DES)})
        self.electrical_cost_dict = {'iteration': list()}
        self.electrical_cost_dict.update({f"des{i}": list() for i in range(n_DES)})
        self.electrical_cost_dict_u = {'iteration': list()}
        self.electrical_cost_dict_u.update({f"des{i}": list() for i in range(n_DES)})
        self.voltage_dict = {f"des{i}": list() for i in range(n_DES)}
        rho = 5e-3 * 4
        inc_rho = 0
        dec_rho = 0
        wait_rho = 3
        mag_pri = 2
        miu = 8
        eta = 2
        counter = 100
        while True:
            rho_u = rho * 5.0
            self.opt_cost_in_des(rho, rho_u)
            # Exchange information with DSO (grid information management system)
            self.exchange_info_in_dso(i)
            tock = time.time() - tick
            dual_error = max([des.dual_error for des in self.des_.values()]) / 1e3  # MW
            pri_error = max([des.pri_error for des in self.des_.values()])/ 1e3  # MW
            if i <= 300:
                if pri_error * mag_pri > dual_error * miu:
                    inc_rho += 1
                    dec_rho = 0
                elif pri_error * 1 < dual_error / miu:
                    inc_rho = 0
                    dec_rho += 0
                else:
                    inc_rho = 0
                    dec_rho = 0
            elif i <= 2500:
                mag_pri = 3.8
                if pri_error * mag_pri > dual_error * miu:
                    inc_rho += 1
                    dec_rho = 0
                elif pri_error * 1 < dual_error / miu:
                    inc_rho = 0
                    dec_rho += 1
                else:
                    inc_rho = 0
                    dec_rho = 0
            else:
                if (i % 100 == 0) and (rho < 2) and (pri_error < 0.15) and (dual_error < 0.15):
                    inc_rho = wait_rho
                    dec_rho = 0
            # cost = sum([des.results["cost"] for des in self.des_.values()])
            obj = sum([des.results["obj"] for des in self.des_.values()])
            # penalty = max([abs(des.results["penalty"]) for des in self.des_.values()])
            self.log_info['iteration'].append(i)
            self.log_info['time'].append(tock)
            self.log_info['rho'].append(rho)
            self.log_info['obj'].append(obj)
            self.log_info['dual_error'].append(dual_error)
            self.log_info['pri_error'].append(pri_error)

            # self.log_info['penalty'].append(penalty)
            self.cost_dict['iteration'].append(i)
            self.electrical_cost_dict['iteration'].append(i)
            self.electrical_cost_dict_u['iteration'].append(i)


            if i > 20:
                obj_recent = np.array(self.log_info['obj'][-15:])
                obj_recent_max = obj_recent.max()
                obj_recent_min = obj_recent.min()
                obj_ptp = obj_recent.ptp()
                if obj_recent.ptp() < 5 and self.log_info['obj'][-1] < obj_recent_max and self.log_info['obj'][
                    -1] > obj_recent_min and rho < 1:
                    inc_rho = wait_rho
                    dec_rho = 0
            else:
                obj_ptp = 1e2

            if i % 1 == 0:
                print(
                    f"Iteration {i}, Time: {tock:.2f} s, rho: {rho}, Dual error: {dual_error} MW, "
                    f"Primal error: {pri_error} MW, Total objective: {obj} "
                    f"inc_rho: {inc_rho}, dec_rho: {dec_rho}, "
                )

            if (dual_error < 1e-5) and (pri_error < 1e-5):
                tick = time.time() - tick
                print(f"Simulation ended, total iterations: {i}, total time: {tick:.2f} seconds")
                print(f"Iteration {i}, max dual error: {dual_error} MW, max primal error: {pri_error} MW")

                break
            if i > 1e4:
                print(
                    f"The simulation has ended, reaching the maximum iteration count of {i} times, with a total time of {tick:. 2f} s")
                break
            i += 1
            counter += 1
            if rho > 100 or rho < 1e-3:
                inc_rho = 0
                dec_rho = 0
            if inc_rho >= wait_rho:
                rho *= eta
                inc_rho = 0
                counter = 0
            if dec_rho >= wait_rho:
                rho *= 1 / eta
                dec_rho = 0


def _map(func, iterable, core_num=None):
    pool = multiprocessing.Pool(2)
    yield from pool.imap_unordered(func, iterable)
    pool.close()


def run_instance(index, des_list, rho):
    results = dict()
    for i in range(index, index + 30):
        des_list[i].update_model(rho)
        results.update({i: {'results': des_list[i].results,
                            'dual_error': des_list[i].dual_error,
                            'pri_error': des_list[i].pri_error,
                            'lambda': des_list[i].lambdas}})
    return results


if not multiprocessing.get_start_method(allow_none=True):
    multiprocessing.set_start_method('spawn')

if __name__ == '__main__':
    network_config_path = "data/6-bus/network_config.json"
    des_config_path = [f"data/6-bus/des{i}_config.json" for i in range(0, n_DES)]
    ss = SimulationSystem(network_config_path, des_config_path)
    ss.run()
