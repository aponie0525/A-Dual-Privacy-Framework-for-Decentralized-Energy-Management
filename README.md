# A-Dual-Privacy-Framework-for-Decentralized-Energy-Management
## Descriptions

We provide scripts for simple simulations of the iterative procedure, which can be run on a single PC without further preparation of the hardwares and their network connections.

This repository contains the Python scripts for simulating the decentralized optimization of a Distributed Energy System (DES). The simulation models the interaction between multiple DES units and a Distribution System Operator (DSO) to achieve optimal energy management in a power network. The system uses an iterative optimization process, ensuring privacy through Random Masking Matrix Encryption (RMME) and homomorphic encryption of exchanged data.

The core logic is divided into the following files:

- **main.py**: The main script to run the entire simulation. It initializes the DSO and all DESs, manages the iterative optimization process, and handles information exchange.

- **dso_lib.py**: Defines the DistributionSystemOperator (DSO) class, which manages the power grid topology (represented as a graph) and facilitates secure data exchange between DES units.

- **des_lib.py**: In the single-PC simulation of the four-layer device-edge-grid-cloud architecture, it defines the DistributedEnergySystem class which consolidates edge-layer functions (local optimization with device constraints/RMME encryption/decryption) and cloud-layer operations (solving RMME-encrypted subproblems), while maintaining cryptographic isolation through CKKS key management and randomized matrix masking to preserve dual-privacy.

- **device_lib.py**: Provides a library of various energy devices (e.g., Solar PV, Wind Power, Batteries, Loads). Each device is defined as a class with its own operational constraints and cost functions.

> **Note**: The simulation reads network and device configurations from JSON and CSV files located in the data/6-bus/ directory. The network configuration file (network_config.json) specifies the topology of the power grid, including the number of nodes, edges, and devices connected to each node. The device configuration file (des{i}_config.json) specifies the operational constraints and cost functions of each DES unit, including the number of devices, their types, and their capacities.

> **Note**: The data for the 60 node distribution grid in Yingkou city is also in the data/ directory.


## System requirements

- **Operating System**: Windows 10 or 11 (64-bit)
- **Processor**: Intel i5 or higher (64-bit)
- **RAM**: Minimum 8GB  
- **Storage**: 2GB free disk space (SSD recommended for faster I/O) 
- **Non-standard hardware**: No additional hardware is required for single-PC simulations.

### Required Software
- **Python**: 3.7 or higher (64-bit)
  - Download from [Python Official Site](https://www.python.org/downloads/)

### Key Dependencies and Versions:
| Package    | Version     | Purpose                            |
|------------|-------------|------------------------------------|
| `cvxpy`    | 1.5.3       | Convex optimization                |
| `osqp`     | 0.6.7.post3 | Quadratic programming solver       |
| `networkx` | 3.2.1       | Graph-based grid topology modeling |
| `numpy`    | 2.3.1       | Numerical operations               |
| `pandas`   | 2.3.1       | Data manipulation                  |
| `scipy`    | 1.16.0      | Scientific computing               |
| `tenseal`  | 0.3.15      | Homomorphic encryption (CKKS)      |

> **Note**: These dependencies ensure cryptographic operations (RMME/CKKS) and optimization solvers (CVXPY) function correctly. Version mismatches may cause errors.


## Installation

The following Python packages are necessary to run the simulation:

- **cvxpy**: For modeling and solving convex optimization problems.

- **numpy**: For numerical operations.

- **pandas**: For data manipulation and reading configuration files.

- **networkx**: For creating and managing the distribution network graph in the DSO.

- **tenseal**: For homomorphic encryption to secure data exchange.

- **osqp**: An efficient solver for quadratic programming problems used by cvxpy.

You can install these packages using pip:

```
pip install -r requirements.txt
```

> **Note**: The `requirements.txt` file contains the list of required packages and their versions.

> **Installation Time**: Approximately 1 minute (varies based on network speed and pip mirror configuration)  

## Demo

To start the simulation of the iteration process, run the `main.py` script from your terminal to simulate the iterative process of a small interconnected distributed system with 6 DESs (data is stored in the `data/6-bus` directory):


python main.py

Due to the memory usage and intensive calculation burden of the simulation on a single PC, the simulation time might be long. In this case, we advise to set nT in des_lib.py and device_lib.py to a smaller value (e.g., 24) to reduce the simulation time and simulate using a dispatch interval (e.g., 1 hour).

This script will:

Initialize the SimulationSystem with the network configuration (data/6-bus/network_config.json) and configurations for the 6 DES units (data/6-bus/des{i}_config.json).

Set up the cryptographic keys for secure communication between the DES units.

Begin the iterative optimization process, where each DES solves its local cost minimization problem and securely exchanges boundary information with its neighbors through the DSO.

The simulation progress, including iteration number, elapsed time, errors, and objective values, will be printed to the console.

A typical output during the simulation will look like this:

```
Simulation ended, total iterations: 279, total time: 92.75 seconds
Iteration 279, max dual error: 9.457416542736287e-06 MW, max primal error: 6.160224516713052e-06 MW...
```

The simulation will stop once the primal and dual errors fall below a predefined tolerance (1e-5) or if the maximum number of iterations is reached. 

> **Running Time**: 92.75s。

## Instructions for use
To run the simulation on a single PC and your own data, follow these steps:
1. Clone the repository to your local machine.
2. Install the required Python packages (cvxpy, numpy, pandas, networkx, tenseal, osqp).
3. Data Preparation: Input the raw data for renewable energy generation, thermal and electric loads, and power grid information into the corresponding CSV files under the `demo/data/6-bus/` directory (e.g., `pv_data.csv`, `wp_data.csv`, etc.).
4. System Configuration Upload: Configure the presence of each type of device for each Distributed Energy System (DES) in the `des_config.csv` file, using `0` to indicate absence and `1` to indicate presence.
5. Generate Configuration Files:
   * Run the `create_json.py` script to automatically generate configuration files for each DES;
   * Run the `create_network_config.py` script to generate the power grid configuration file;
   * If you wish to modify any device parameters, you may manually edit the corresponding configuration file located at `data/6-bus/des{i}_config.json`.
6. Start the Simulation: Execute the main script `main.py` to initiate the simulation and optimization process for the interconnected distributed energy systems.
For further customization or debugging, please ensure that all data files are formatted correctly and adjust configuration parameters as needed.
> **Note**: The simulation reads network and device configurations from JSON and CSV files located in the data/ directory. The network configuration file (network_config.json) specifies the topology of the power grid, including the number of nodes, edges, and devices connected to each node. The device configuration file (des{i}_config.json) specifies the operational constraints and cost functions of each DES unit, including the number of devices, their types, and their capacities.

## Other information

There are other parameters in the codes that can be set for further simulations. Please refer to the source code and data.

Contact us if you have further questions on deployment of the code: chenqun@tsinghua.edu.cn, mahuaneea@outlook.com
