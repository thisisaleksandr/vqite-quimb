import numpy as np
import pickle
import time
from tqdm import tqdm
from typing import (
    List,
    Optional,
    Tuple,
    Union
)

import quimb as qu
import quimb.tensor as qtn

import cotengra as ctg


def add_pauli_rotation_gate(
    qc: "quimb.tensor.circuit.Circuit",
    pauli_string: str,
    theta: float,
    decompose_rzz: bool = True
):
    """
    Appends a Pauli rotation gate to a Quimb Circuit.
    Convention for Pauli string ordering is opposite to the Qiskit convention.
    For example, in string "XYZ" Pauli "X" acts on the first qubit.

    Parameters
    ----------
    qc : "quimb.tensor.circuit.Circuit"
        Quimb Circuit to which the Pauli rotation gate is appended.
    pauli_string : str
        Pauli string defining the rotation.
    theta : float
        Rotation angle.
    decompose_rzz : bool
        If decompose_rzz==True, all rzz gates are decompsed into cx-rz-cx.
        Otherwise, the final circuit contains rzz gates.

    Returns
    -------
    qc: Parameterized "quimb.tensor.circuit.Circuit"
    """

    if qc.N != len(pauli_string):
        raise ValueError("Circuit and Pauli string are of different size")
    if all([pauli=='I' or pauli=='X' or pauli=='Y' or pauli=='Z'
            for pauli in pauli_string])==False:
        raise ValueError("Pauli string does not have a correct format")

    nontriv_pauli_list = [(i,pauli)
                        for i,pauli in enumerate(pauli_string) if pauli!='I']

    if len(nontriv_pauli_list)==1:
        if nontriv_pauli_list[0][1]=='X':
            qc.apply_gate('RX',theta,nontriv_pauli_list[0][0],parametrize=True)
        if nontriv_pauli_list[0][1]=='Y':
            qc.apply_gate('RY',theta,nontriv_pauli_list[0][0],parametrize=True)
        if nontriv_pauli_list[0][1]=='Z':
            qc.apply_gate('RZ',theta,nontriv_pauli_list[0][0],parametrize=True)
    elif len(nontriv_pauli_list)==2 and nontriv_pauli_list[0][1]+nontriv_pauli_list[1][1] == 'XX':
            qc.apply_gate('RXX',theta,nontriv_pauli_list[0][0],nontriv_pauli_list[1][0],parametrize=True)
    elif len(nontriv_pauli_list)==2 and nontriv_pauli_list[0][1]+nontriv_pauli_list[1][1] == 'YY':
            qc.apply_gate('RYY',theta,nontriv_pauli_list[0][0],nontriv_pauli_list[1][0],parametrize=True)
    else:
        for (i,pauli) in nontriv_pauli_list:
            if pauli=='X':
                qc.apply_gate('H',i)
            if pauli=='Y':
                qc.apply_gate('SDG',i)
                qc.apply_gate('H',i)
        for list_ind in range(len(nontriv_pauli_list)-2):
            qc.apply_gate('CX',nontriv_pauli_list[list_ind][0],nontriv_pauli_list[list_ind+1][0])
        if decompose_rzz==True:
            qc.apply_gate(
                'CX',
                nontriv_pauli_list[len(nontriv_pauli_list)-2][0],
                nontriv_pauli_list[len(nontriv_pauli_list)-1][0]
            )
            qc.apply_gate('RZ',theta,nontriv_pauli_list[len(nontriv_pauli_list)-1][0],parametrize=True)
            qc.apply_gate(
                'CX',
                nontriv_pauli_list[len(nontriv_pauli_list)-2][0],
                nontriv_pauli_list[len(nontriv_pauli_list)-1][0]
            )
        if decompose_rzz==False:
            qc.apply_gate(
                'RZZ',
                theta,
                nontriv_pauli_list[len(nontriv_pauli_list)-2][0],
                nontriv_pauli_list[len(nontriv_pauli_list)-1][0],
                parametrize=True
            )
        for list_ind in reversed(range(len(nontriv_pauli_list)-2)):
            qc.apply_gate('CX',nontriv_pauli_list[list_ind][0],nontriv_pauli_list[list_ind+1][0])
        for (i,pauli) in nontriv_pauli_list:
            if pauli=='X':
                qc.apply_gate('H',i)
            if pauli=='Y':
                qc.apply_gate('H',i)
                qc.apply_gate('S',i)
    return qc


class model_H:
    def __init__(
        self,
        incar_file: str
    ):
        with open(incar_file) as fp:
            incar_content = fp.read()

        h_pos = incar_content.find("h")
        pool_pos = incar_content.find("pool")
        h_string = incar_content[h_pos+14:pool_pos-14]

        self.paulis = "".join([el for el in h_string if el=='I' or el=='X' or el=='Y' or el=='Z' or el == '\n']).split('\n')

        coefs_str = "".join([el for el in h_string if el.isdigit() or el=="-" or el=="." or el == "*"]).split('*')
        self.coefs = [float(el) for el in coefs_str[0:-1]]


class Quimb_avqite_contractions_estimates:
    def __init__(
        self,
        num_qubits: int,
        g: float,
        filename: str
    ):
        self._num_qubits = num_qubits
        self._g = g
        self._filename = filename

        self._H = model_H("incars/incar"+self._filename)

        #Reads out the ansatz file.
        (self._ansatz_adaptvqite,
         self._params_ansatz) = self.read_adaptvqite_ansatz("adaptvqite/adaptvqite/data/ansatz_inp.pkle")

        #Reads out the incar file.
        with open("incars/incar"+self._filename) as fp:
            incar_content = fp.read()
        ref_st_r_pos = incar_content.find("ref_state")
        #Reads out the reference state from the incar file.
        self._ref_state = incar_content[
                            ref_st_r_pos+13:ref_st_r_pos+13+self._num_qubits
                            ]

        #Initializes a QuantumCircuit object.
        self._init_qc = qtn.Circuit(N=self._num_qubits)

        #If the reference state contains "1"s, adds corresponding bit-flips.
        if all([(el=='0') or (el=='1') for el in self._ref_state]):
            [self._init_qc.apply_gate('X',i) for i,el in enumerate(self._ref_state) if el=='1']
        else:
            raise ValueError(
                "Reference state is supposed to be a string of 0s and 1s"
            )

        self.pauli_rot_gates_list = [add_pauli_rotation_gate(qc=qtn.Circuit(N=self._num_qubits),pauli_string=self._ansatz_adaptvqite[i],theta=self._params_ansatz[i],decompose_rzz=False).gates for i in range(len(self._ansatz_adaptvqite))]

        self.pauli_rot_dag_gates_list = [add_pauli_rotation_gate(qc=qtn.Circuit(N=self._num_qubits),pauli_string=self._ansatz_adaptvqite[i],theta=-self._params_ansatz[i],decompose_rzz=False).gates for i in range(len(self._ansatz_adaptvqite))]

        self.base_circuits = [self.circuit_2(mu) for mu in range(len(self._ansatz_adaptvqite)+1)]


    def read_adaptvqite_ansatz(
        self,
        filename: str
    ):
        """
        Reads the ansatz from a file resulting from adaptvqite calculation.

        Parameters
        ----------
        filename : str
            Name of a file containing the results of adaptvqite calculation.
            Has to be given in .pkle format.

        Returns
        -------
        ansatz_adaptvqite : List[str]
            List of Pauli strings entering the ansatz.
        params_adaptvqite : List[float64]
            Parameters (angles) of the ansatz.
        """
        if filename[-5:] != '.pkle':
            raise ImportError("Ansatz file should be given in .pkle format")

        with open(filename, 'rb') as inp:
            data_inp = pickle.load(inp)
            ansatz_adaptvqite = data_inp[0]
            params_adaptvqite = data_inp[1]
            # params_adaptvqite = list(np.random.random(20))

        return ansatz_adaptvqite, params_adaptvqite


    def pauli_string_to_quimb_gates(
        self,
        pauli_string
    ):
        gates = ()
        for i,el in enumerate(pauli_string):
            if el=='X':
                gates = gates + (qtn.circuit.Gate(label='X', params=[], qubits=(i,)),)
            if el=='Y':
                gates = gates + (qtn.circuit.Gate(label='Y', params=[], qubits=(i,)),)
            if el=='Z':
                gates = gates + (qtn.circuit.Gate(label='Z', params=[], qubits=(i,)),)
        return gates


    def circuit_1(
        self,
        mu: int,
        nu: int,
        A_mu: Union["Instruction", "Operator"],
        A_nu: Union["Instruction", "Operator"]
    ):
        if mu >= nu:
            raise ValueError("Here mu<nu is required.")
        if mu > len(self._ansatz_adaptvqite) or nu > len(self._ansatz_adaptvqite):
            raise ValueError("mu, nu has to be smaller than the number of operators in the ansatz")

        qc = self.base_circuits[mu].copy()


        qc.apply_gates( self.pauli_string_to_quimb_gates(pauli_string=A_mu), contract=False )
        for i in range(mu,nu):
             qc.apply_gates(self.pauli_rot_gates_list[i], contract=False)
        qc.apply_gates( self.pauli_string_to_quimb_gates(pauli_string=A_nu), contract=False )
        for i in reversed(range(nu)):
            qc.apply_gates(self.pauli_rot_dag_gates_list[i], contract=False)

        return qc


    def circuit_2(
        self,
        mu: int
    ):
        qc = self._init_qc.copy()

        for i in range(mu):
            qc.apply_gates(self.pauli_rot_gates_list[i], contract=False)

        return qc


    def quimb_ampl_contr_est(
        self,
        circuit,
        opt = 'hyper'
    ):
        t3 = time.time()
        reh = circuit.amplitude_rehearse('0'*self._num_qubits, optimize=opt)
        t4 = time.time()
        width, cost = reh['tree'].contraction_width(), reh['tree'].contraction_cost()
        t5 = time.time()
        contraction = reh['tn'].contract(all, optimize=reh['tree'], output_inds=())
        t6 = time.time()

        # print(t4-t3,t5-t4,t6-t5)
        return width, cost, contraction


    def quimb_local_exp_contr_est(
        self,
        circuit,
        operator,
        where,
        opt = 'hyper'
    ):
        reh = circuit.local_expectation_rehearse(operator, where, optimize=opt)

        width, cost = reh['tree'].contraction_width(), reh['tree'].contraction_cost()

        contraction = reh['tn'].contract(all, optimize=reh['tree'], output_inds=())

        return width, cost, contraction


    def h_exp_value(
        self,
        params,
        opt = None
    ):

        opt_reuse = ctg.ReusableHyperOptimizer(
            max_repeats=16,
            reconf_opts={},
            parallel=False,
            progbar=False,
        #     directory=True,  # if you want a persistent path cache
        )

        if opt==None:
            opt='greedy'

        qc = self.base_circuits[-1].copy()

        old_params_dict = qc.get_params()
        new_params_dict = dict()
        for i,key in enumerate(old_params_dict.keys()):
            new_params_dict[key]= np.array([params[i]])

        qc.set_params(new_params_dict)

        h_exp_vals = []



        for pauli_string in self._H.paulis:
            where = [i for i,p in enumerate(pauli_string) if p!= 'I']
            paulis = [p for i,p in enumerate(pauli_string) if p!= 'I']

            operator = qu.pauli(paulis[0])
            for i in range(1,len(where)):
                operator = operator & qu.pauli(paulis[i])

            width, cost, contraction = self.quimb_local_exp_contr_est(circuit = qc, operator = operator, where = where, opt=opt)

            h_exp_vals.append(contraction)

        exp_value = sum([h_exp_vals[i]*self._H.coefs[i] for i in range(len(self._H.coefs))])

        return exp_value



    def avqite_contr_est(
        self,
        contr_type: int,
        mu: Optional[int] = None,
        nu: Optional[int] = None,
    ):
        if contr_type == 1 and type(mu) != int and type(nu) != int:
            raise ValueError("For contraction type 1, both mu and nu have to be int")
        if contr_type == 2 and type(mu) != int and nu != None:
            raise ValueError("For contraction type 2, mu has to be int and nu has to be None")
        if contr_type == 3 and type(mu) != int and nu != None:
            raise ValueError("For contraction type 3, mu has to be int and nu has to be None")
        if contr_type == 4 and mu != None and nu != None:
            raise ValueError("For contraction type 4, mu has to be None and nu has to be None")
        if contr_type == 5 and mu != None and nu != None:
            raise ValueError("For contraction type 5, mu has to be None and nu has to be None")

        if mu != None and nu != None:
            if mu > len(self._ansatz_adaptvqite) or nu > len(self._ansatz_adaptvqite):
                raise ValueError("mu, nu has to be smaller than the number of operators in the ansatz")

        if contr_type == 1:
            if mu<nu:
                t1=time.time()
                qc = self.circuit_1(mu, nu, A_mu = self._ansatz_adaptvqite[mu], A_nu = self._ansatz_adaptvqite[nu])
                t2=time.time()
                width, cost, contraction = self.quimb_ampl_contr_est(circuit = qc)
                t3=time.time()
                # print(t2-t1,t3-t2)
            if mu==nu:
                width, cost, contraction = (1,0,1)
            contraction = np.real(contraction)/4

        if contr_type == 2:
            qc = self.base_circuits[mu]
            where = [i for i,p in enumerate(self._ansatz_adaptvqite[mu]) if p!= 'I']
            paulis = [p for i,p in enumerate(self._ansatz_adaptvqite[mu]) if p!= 'I']

            operator = qu.pauli(paulis[0])
            for i in range(1,len(where)):
                operator = operator & qu.pauli(paulis[i])

            width, cost, contraction = self.quimb_local_exp_contr_est(circuit = qc, operator = operator, where = where)
            contraction = np.real(1j*contraction/2)

        return width, cost, contraction
