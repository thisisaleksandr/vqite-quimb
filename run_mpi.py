import numpy as np
import scipy
import pickle
import time
from tqdm import tqdm
from typing import (
    List,
    Optional,
    Tuple,
    Union
)
import random
import importlib
from mpi4py import MPI

import quimb as qu
import quimb.tensor as qtn

import cotengra as ctg

import avqite_quimb_mpi as avqite_quimb


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank==0:
    avqite_quimb_obj = avqite_quimb.Quimb_vqite(
        incar_file = "adaptvqite/adaptvqite/incars/incarN12g0.1",
        ansatz_file = "adaptvqite/adaptvqite/data/ansatz_inp_i20.pkle",
        init_params = "avqite"
    )
    init_params = avqite_quimb_obj._params
else:
    init_params = None

init_params = comm.bcast(init_params, root=0)

if rank!=0:
    avqite_quimb_obj = avqite_quimb.Quimb_vqite(
        incar_file = "adaptvqite/adaptvqite/incars/incarN12g0.1",
        ansatz_file = "adaptvqite/adaptvqite/data/ansatz_inp_i20.pkle",
        init_params = init_params
    )

with open("adaptvqite/adaptvqite/data/M_V_i20.pkle", 'rb') as inp:
    data_inp = pickle.load(inp)
    M_adaptvqite = data_inp[0]
    V_adaptvqite = data_inp[1]

# which_nonzero=[(0, 0),
#  (0, 16),
#  (1, 1),
#  (1, 19),
#  (2, 2),
#  (2, 17),
#  (3, 3),
#  (3, 11),
#  (3, 20),
#  (4, 4),
#  (5, 5),
#  (5, 12),
#  (5, 17),
#  (6, 6),
#  (6, 13),
#  (7, 7),
#  (8, 8),
#  (8, 15),
#  (9, 9),
#  (9, 18),
#  (10, 10),
#  (11, 11),
#  (11, 20),
#  (12, 12),
#  (13, 13),
#  (14, 14),
#  (15, 15),
#  (15, 16),
#  (16, 16),
#  (17, 17),
#  (18, 18),
#  (18, 19),
#  (19, 19),
#  (20, 20)]
# start_time = MPI.Wtime()
# avqite_quimb_obj.compute_m(which_nonzero=None,optimize='greedy',simplify_sequence = '',backend=None)
# end_time = MPI.Wtime()
# Mdiff = M_adaptvqite - avqite_quimb_obj._m
# if rank==0:
#     print("time: ",end_time-start_time)
#     print(np.where((Mdiff>1e-14) == True))

# start_time = MPI.Wtime()
# avqite_quimb_obj.compute_v(optimize='greedy',simplify_sequence = '',backend=None)
# end_time = MPI.Wtime()
# Vdiff = V_adaptvqite - avqite_quimb_obj._v
# if rank==0:
#     print("time: ",end_time-start_time)
#     print(np.where((Vdiff>1e-14) == True))


if rank==0:
    with open("output.txt", "w") as f:
        print(avqite_quimb_obj._params, file=f)
    
avqite_quimb_obj.vqite(optimize_m='greedy',optimize_v='greedy',simplify_sequence = '',backend=None)

if rank==0:
    with open("output.txt", "a") as f:
        print("Final energy:", avqite_quimb_obj._e, avqite_quimb_obj._params, file=f)

