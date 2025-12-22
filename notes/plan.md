

function retunrs:



Dict per configuration:
    C_tilde_DD - Non-Maxwell Capacitance
    C_DG - Gate-dot capacitance matrix
    tc - Tunnel couplings
    v_offset - Offset voltage for the gate
    "cut"::
       '001-100':
        x_voltage - [x0,x1,N] linspace, taking into account the sizes
        y_voltage - [y0,y1,N] linspace, taking into account the sizes
        
    


C_tilde_DD  -> array(batch_size, Nd, Nd)
C_DG -> array(batch_size, Nd, Ng)
tc -> array(batch_size, Nd, Nd)
v_offset -> array(batch_size, Ng)


x_voltage -> array(batch_size, Ncuts, 3)
y_voltage -> array(batch_size, Ncuts, 3)
lut_cuts -> ['100-100':0, '001-110':1, ... ]



data['cut]['voltage']

    



## Issues
### Data generation
Issue: while generating 5000 datapoints, got errors for 20 realizations. 

All folders' IDs with errors: 03962, 01711, 04201, 00035, 01613, 00361, 00191, 00940, 00985, 03613, 03419, 03087, 03864, 01159, 04229, 02549, 04708, 04790, 00195, 00569.

Not all are categorized yet. Last one 3613.

#### Example 1:
The same error happens for the realizations (folder ID): 4790, 4201
Error calls for few are printed below.
```
Generating realization 4790/5000...
EXPERIMENT INITIALIZED
-----------------------

                Capacitance model deployed with the following parameters:
                Dot-dot capacitances: 
                [[17.83513076  4.22090325  3.03910431  1.57198737]
 [ 4.22090325 20.69445648  2.19203752  0.89565073]
 [ 3.03910431  2.19203752 26.78339442  1.32287131]
 [ 1.57198737  0.89565073  1.32287131 13.87704303]]
                Dot-gate capacitances: 
                [[ 7.29602798  0.54665757  0.37228031  0.        ]
 [ 0.89820057 11.14375961  0.22846668  0.        ]
 [ 0.45137979  0.33690593 13.51511488  0.        ]
 [ 0.          0.          0.          5.        ]]
                Size of Coulomb peaks V[n] is constant
                

            Sensor model deployed with the following parameters:   
            Sensor dot indices: [3]
            Sensor detunings: [1000.] meV
            Coulomb peak width: 2.18 meV
            Slow noise amplitude: 0.0 ueV
            Fast noise amplitude: 0.0 ueV
            Signal noise scale: 0.0
            

            Tunneling model deployed with the following parameters:
            Tunneling matrix:
            [[0.00000000e+00 6.67718926e-05 2.27460059e-05 0.00000000e+00]
 [6.67718926e-05 0.00000000e+00 2.23619522e-06 0.00000000e+00]
 [2.27460059e-05 2.23619522e-06 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]]
            Temperature: 0.1 K
            Energy range factor: 3
            
Rastering CSD: 100%|██████████| 22500/22500 [00:00<00:00, 35684.15it/s]
Rastering CSD: 100%|██████████| 22500/22500 [00:00<00:00, 38890.13it/s]
Rastering CSD: 100%|██████████| 22500/22500 [00:00<00:00, 38129.24it/s]
Traceback (most recent call last):
  File "/zfsstore/user/kreftb/QD-CNN/src/utilities/utils.py", line 522, in generate_datapoint
    base_state = cap_sim.find_state_of_voltage(np.zeros(cap_sim.num_dots), state_hint=base_state_hint)
  File "/zfsstore/user/kreftb/QD-CNN/qdarts/simulator.py", line 360, in find_state_of_voltage
    state, v_inside = self.find_boundary_intersection(
  File "/zfsstore/user/kreftb/QD-CNN/qdarts/simulator.py", line 274, in find_boundary_intersection
    raise ValueError("old_v must be in the provided state.")
ValueError: old_v must be in the provided state.
Generated datapoint 04789 with 3 cuts
```
```
Generating realization 4201/5000...
EXPERIMENT INITIALIZED
-----------------------

                Capacitance model deployed with the following parameters:
                Dot-dot capacitances: 
                [[24.32972975  6.14400886  1.98738826  1.20379114]
 [ 6.14400886 27.51337315  3.48694098  0.65542199]
 [ 1.98738826  3.48694098 18.91417712  1.60866115]
 [ 1.20379114  0.65542199  1.60866115 13.16219807]]
                Dot-gate capacitances: 
                [[9.97580359 0.68529987 0.23443934 0.        ]
 [0.28548664 9.84921498 0.32279767 0.        ]
 [0.15005553 0.47768157 9.75879423 0.        ]
 [0.         0.         0.         5.        ]]
                Size of Coulomb peaks V[n] is constant
                

            Sensor model deployed with the following parameters:   
            Sensor dot indices: [3]
            Sensor detunings: [1000.] meV
            Coulomb peak width: 2.18 meV
            Slow noise amplitude: 0.0 ueV
            Fast noise amplitude: 0.0 ueV
            Signal noise scale: 0.0
            

            Tunneling model deployed with the following parameters:
            Tunneling matrix:
            [[0.00000000e+00 8.40478287e-05 2.59128749e-07 0.00000000e+00]
 [8.40478287e-05 0.00000000e+00 1.26143728e-05 0.00000000e+00]
 [2.59128749e-07 1.26143728e-05 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]]
            Temperature: 0.1 K
            Energy range factor: 3
            
Rastering CSD: 100%|██████████| 22500/22500 [00:00<00:00, 33903.96it/s]
Rastering CSD: 100%|██████████| 22500/22500 [00:00<00:00, 27037.84it/s]
Rastering CSD: 100%|██████████| 22500/22500 [00:00<00:00, 35626.56it/s]
Traceback (most recent call last):
  File "/zfsstore/user/kreftb/QD-CNN/src/utilities/utils.py", line 522, in generate_datapoint
    base_state = cap_sim.find_state_of_voltage(np.zeros(cap_sim.num_dots), state_hint=base_state_hint)
  File "/zfsstore/user/kreftb/QD-CNN/qdarts/simulator.py", line 360, in find_state_of_voltage
    state, v_inside = self.find_boundary_intersection(
  File "/zfsstore/user/kreftb/QD-CNN/qdarts/simulator.py", line 274, in find_boundary_intersection
    raise ValueError("old_v must be in the provided state.")
ValueError: old_v must be in the provided state.
Generated datapoint 04200 with 3 cuts
```


#### Example 2:
The same error happens for the realizations (folder ID): 4708, 3962, 1711, 361, 3613, 
Error calls for few are printed below.

```
Generating realization 4708/5000...
EXPERIMENT INITIALIZED
-----------------------

                Capacitance model deployed with the following parameters:
                Dot-dot capacitances: 
                [[22.00863651  5.33133196  4.13780542  1.81853424]
 [ 5.33133196 18.85743695  0.28585756  0.85426842]
 [ 4.13780542  0.28585756 20.65653476  2.01927281]
 [ 1.81853424  0.85426842  2.01927281 15.82803454]]
                Dot-gate capacitances: 
                [[5.81470392e+00 2.78419239e-01 1.65547649e-01 0.00000000e+00]
 [2.04342985e-01 5.55062094e+00 2.04520997e-03 0.00000000e+00]
 [3.49275678e-01 4.75416685e-02 8.74185005e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 5.00000000e+00]]
                Size of Coulomb peaks V[n] is constant
                

            Sensor model deployed with the following parameters:   
            Sensor dot indices: [3]
            Sensor detunings: [1000.] meV
            Coulomb peak width: 2.18 meV
            Slow noise amplitude: 0.0 ueV
            Fast noise amplitude: 0.0 ueV
            Signal noise scale: 0.0
            

            Tunneling model deployed with the following parameters:
            Tunneling matrix:
            [[0.00000000e+00 1.59055899e-05 5.27490047e-05 0.00000000e+00]
 [1.59055899e-05 0.00000000e+00 1.48797539e-08 0.00000000e+00]
 [5.27490047e-05 1.48797539e-08 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]]
            Temperature: 0.1 K
            Energy range factor: 3
            
Rastering CSD: 100%|██████████| 22500/22500 [00:00<00:00, 35829.11it/s]
Rastering CSD: 100%|██████████| 22500/22500 [00:00<00:00, 37308.44it/s]
Rastering CSD: 100%|██████████| 22500/22500 [00:00<00:00, 33945.47it/s]
Traceback (most recent call last):
  File "/zfsstore/user/kreftb/QD-CNN/src/utilities/utils.py", line 556, in generate_datapoint
    xout, yout, _, polytopes, sensor_values, _ = experiment.generate_CSD(
  File "/zfsstore/user/kreftb/QD-CNN/qdarts/experiment.py", line 522, in generate_CSD
    backend, CSD_data, states = get_CSD_data(
  File "/zfsstore/user/kreftb/QD-CNN/qdarts/plotting.py", line 141, in get_CSD_data
    corner_state = simulation.find_state_of_voltage(
  File "/zfsstore/user/kreftb/QD-CNN/qdarts/simulator.py", line 360, in find_state_of_voltage
    state, v_inside = self.find_boundary_intersection(
  File "/zfsstore/user/kreftb/QD-CNN/qdarts/simulator.py", line 287, in find_boundary_intersection
    transition_idx = np.argmin(ts)
  File "/zfsstore/user/kreftb/QD-CNN/env/lib64/python3.9/site-packages/numpy/_core/fromnumeric.py", line 1395, in argmin
    return _wrapfunc(a, 'argmin', axis=axis, out=out, **kwds)
  File "/zfsstore/user/kreftb/QD-CNN/env/lib64/python3.9/site-packages/numpy/_core/fromnumeric.py", line 57, in _wrapfunc
    return bound(*args, **kwds)
ValueError: attempt to get argmin of an empty sequence
Generated datapoint 04707 with 3 cuts
```
```
Generating realization 3962/5000...
EXPERIMENT INITIALIZED
-----------------------

                Capacitance model deployed with the following parameters:
                Dot-dot capacitances: 
                [[22.24512382  3.03217674  2.39516048  1.93670071]
 [ 3.03217674 25.73027171  2.77735353  0.84255458]
 [ 2.39516048  2.77735353 22.06236787  0.64063121]
 [ 1.93670071  0.84255458  0.64063121 12.99816996]]
                Dot-gate capacitances: 
                [[6.67926234 0.31367838 0.07079713 0.        ]
 [0.55039606 8.22586902 0.40562921 0.        ]
 [0.13830547 0.45411643 9.68740484 0.        ]
 [0.         0.         0.         5.        ]]
                Size of Coulomb peaks V[n] is constant
                

            Sensor model deployed with the following parameters:   
            Sensor dot indices: [3]
            Sensor detunings: [1000.] meV
            Coulomb peak width: 2.18 meV
            Slow noise amplitude: 0.0 ueV
            Fast noise amplitude: 0.0 ueV
            Signal noise scale: 0.0
            

            Tunneling model deployed with the following parameters:
            Tunneling matrix:
            [[0.00000000e+00 5.21389057e-05 1.75397632e-07 0.00000000e+00]
 [5.21389057e-05 0.00000000e+00 3.91131904e-05 0.00000000e+00]
 [1.75397632e-07 3.91131904e-05 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]]
            Temperature: 0.1 K
            Energy range factor: 3
            
Rastering CSD: 100%|██████████| 22500/22500 [00:00<00:00, 32099.10it/s]
Rastering CSD: 100%|██████████| 22500/22500 [00:00<00:00, 38160.80it/s]
Rastering CSD: 100%|██████████| 22500/22500 [00:00<00:00, 32894.16it/s]
Traceback (most recent call last):
  File "/zfsstore/user/kreftb/QD-CNN/src/utilities/utils.py", line 556, in generate_datapoint
    xout, yout, _, polytopes, sensor_values, _ = experiment.generate_CSD(
  File "/zfsstore/user/kreftb/QD-CNN/qdarts/experiment.py", line 522, in generate_CSD
    backend, CSD_data, states = get_CSD_data(
  File "/zfsstore/user/kreftb/QD-CNN/qdarts/plotting.py", line 141, in get_CSD_data
    corner_state = simulation.find_state_of_voltage(
  File "/zfsstore/user/kreftb/QD-CNN/qdarts/simulator.py", line 360, in find_state_of_voltage
    state, v_inside = self.find_boundary_intersection(
  File "/zfsstore/user/kreftb/QD-CNN/qdarts/simulator.py", line 287, in find_boundary_intersection
    transition_idx = np.argmin(ts)
  File "/zfsstore/user/kreftb/QD-CNN/env/lib64/python3.9/site-packages/numpy/_core/fromnumeric.py", line 1395, in argmin
    return _wrapfunc(a, 'argmin', axis=axis, out=out, **kwds)
  File "/zfsstore/user/kreftb/QD-CNN/env/lib64/python3.9/site-packages/numpy/_core/fromnumeric.py", line 57, in _wrapfunc
    return bound(*args, **kwds)
ValueError: attempt to get argmin of an empty sequence
Generated datapoint 03961 with 3 cuts
```

```
Generating realization 1711/5000...
EXPERIMENT INITIALIZED
-----------------------

                Capacitance model deployed with the following parameters:
                Dot-dot capacitances: 
                [[21.42685855  2.15276959  2.0876244   0.74711495]
 [ 2.15276959 22.21058095  4.21578031  1.31900027]
 [ 2.0876244   4.21578031 23.91979072  2.17825452]
 [ 0.74711495  1.31900027  2.17825452 14.70186084]]
                Dot-gate capacitances: 
                [[8.21530705 0.42286461 0.09059217 0.        ]
 [0.49289403 8.835093   0.36080501 0.        ]
 [0.04330781 0.54178085 9.83738237 0.        ]
 [0.         0.         0.         5.        ]]
                Size of Coulomb peaks V[n] is constant
                

            Sensor model deployed with the following parameters:   
            Sensor dot indices: [3]
            Sensor detunings: [1000.] meV
            Coulomb peak width: 2.18 meV
            Slow noise amplitude: 0.0 ueV
            Fast noise amplitude: 0.0 ueV
            Signal noise scale: 0.0
            

            Tunneling model deployed with the following parameters:
            Tunneling matrix:
            [[0.00000000e+00 2.15529761e-05 2.55878942e-08 0.00000000e+00]
 [2.15529761e-05 0.00000000e+00 2.93663827e-05 0.00000000e+00]
 [2.55878942e-08 2.93663827e-05 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]]
            Temperature: 0.1 K
            Energy range factor: 3
            
Rastering CSD: 100%|██████████| 22500/22500 [00:00<00:00, 35052.55it/s]
Rastering CSD: 100%|██████████| 22500/22500 [00:00<00:00, 38002.85it/s]
Rastering CSD: 100%|██████████| 22500/22500 [00:00<00:00, 37240.71it/s]
Traceback (most recent call last):
  File "/zfsstore/user/kreftb/QD-CNN/src/utilities/utils.py", line 522, in generate_datapoint
    base_state = cap_sim.find_state_of_voltage(np.zeros(cap_sim.num_dots), state_hint=base_state_hint)
  File "/zfsstore/user/kreftb/QD-CNN/qdarts/simulator.py", line 360, in find_state_of_voltage
    state, v_inside = self.find_boundary_intersection(
  File "/zfsstore/user/kreftb/QD-CNN/qdarts/simulator.py", line 274, in find_boundary_intersection
    raise ValueError("old_v must be in the provided state.")
ValueError: old_v must be in the provided state.
Generated datapoint 01710 with 3 cuts
Generating realization 1712/5000...
```

Also there might be some issue with saving and indexing.
```
Generating realization 192/5000...
EXPERIMENT INITIALIZED
-----------------------

                Capacitance model deployed with the following parameters:
                Dot-dot capacitances: 
                [[23.74682668  3.28420717  4.10799778  0.97001697]
 [ 3.28420717 20.07124525  2.04267988  0.82242345]
 [ 4.10799778  2.04267988  6.9456346   1.29196322]
 [ 0.97001697  0.82242345  1.29196322 13.98581915]]
                Dot-gate capacitances: 
                [[6.01606096 0.09654062 0.33508175 0.        ]
 [0.32369933 7.02248673 0.04770626 0.        ]
 [0.18041122 0.02580882 0.         0.        ]
 [0.         0.         0.         5.        ]]
                Size of Coulomb peaks V[n] is constant
                

            Sensor model deployed with the following parameters:   
            Sensor dot indices: [3]
            Sensor detunings: [1000.] meV
            Coulomb peak width: 2.18 meV
            Slow noise amplitude: 0.0 ueV
            Fast noise amplitude: 0.0 ueV
            Signal noise scale: 0.0
            

            Tunneling model deployed with the following parameters:
            Tunneling matrix:
            [[0.00000000e+00 1.00807873e-05 9.00507885e-05 0.00000000e+00]
 [1.00807873e-05 0.00000000e+00 1.94471985e-08 0.00000000e+00]
 [9.00507885e-05 1.94471985e-08 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]]
            Temperature: 0.1 K
            Energy range factor: 3
            
Rastering CSD: 100%|██████████| 22500/22500 [00:00<00:00, 38658.83it/s]
Traceback (most recent call last):
  File "/zfsstore/user/kreftb/QD-CNN/src/utilities/utils.py", line 556, in generate_datapoint
    xout, yout, _, polytopes, sensor_values, _ = experiment.generate_CSD(
  File "/zfsstore/user/kreftb/QD-CNN/qdarts/experiment.py", line 522, in generate_CSD
    backend, CSD_data, states = get_CSD_data(
  File "/zfsstore/user/kreftb/QD-CNN/qdarts/plotting.py", line 141, in get_CSD_data
    corner_state = simulation.find_state_of_voltage(
  File "/zfsstore/user/kreftb/QD-CNN/qdarts/simulator.py", line 360, in find_state_of_voltage
    state, v_inside = self.find_boundary_intersection(
  File "/zfsstore/user/kreftb/QD-CNN/qdarts/simulator.py", line 287, in find_boundary_intersection
    transition_idx = np.argmin(ts)
  File "/zfsstore/user/kreftb/QD-CNN/env/lib64/python3.9/site-packages/numpy/_core/fromnumeric.py", line 1395, in argmin
    return _wrapfunc(a, 'argmin', axis=axis, out=out, **kwds)
  File "/zfsstore/user/kreftb/QD-CNN/env/lib64/python3.9/site-packages/numpy/_core/fromnumeric.py", line 57, in _wrapfunc
    return bound(*args, **kwds)
ValueError: attempt to get argmin of an empty sequence
Error generating datapoint 00191: attempt to get argmin of an empty sequence
Generating realization 193/5000...
```


#### Example 3:
The same error happens for the realizations (folder ID): 35, 
Error calls for few are printed below.

```
Generating realization 35/5000...
EXPERIMENT INITIALIZED
-----------------------

                Capacitance model deployed with the following parameters:
                Dot-dot capacitances: 
                [[16.9301838   3.59219049  4.14521071  1.70868403]
 [ 3.59219049 14.26153674  3.6407325   0.96826758]
 [ 4.14521071  3.6407325  16.01879469  0.43335747]
 [ 1.70868403  0.96826758  0.43335747 15.2223302 ]]
                Dot-gate capacitances: 
                [[5.58184855 0.47412416 0.47156713 0.        ]
 [0.56071344 6.78990414 0.14315695 0.        ]
 [0.34888418 0.1842462  6.05030806 0.        ]
 [0.         0.         0.         5.        ]]
                Size of Coulomb peaks V[n] is constant
                

            Sensor model deployed with the following parameters:   
            Sensor dot indices: [3]
            Sensor detunings: [1000.] meV
            Coulomb peak width: 2.18 meV
            Slow noise amplitude: 0.0 ueV
            Fast noise amplitude: 0.0 ueV
            Signal noise scale: 0.0
            

            Tunneling model deployed with the following parameters:
            Tunneling matrix:
            [[0.00000000e+00 8.37500067e-05 3.76714813e-05 0.00000000e+00]
 [8.37500067e-05 0.00000000e+00 1.29442702e-05 0.00000000e+00]
 [3.76714813e-05 1.29442702e-05 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]]
            Temperature: 0.1 K
            Energy range factor: 3
            
Rastering CSD: 100%|██████████| 22500/22500 [00:00<00:00, 33776.20it/s]
Rastering CSD: 100%|██████████| 22500/22500 [00:00<00:00, 37585.12it/s]
Rastering CSD: 100%|██████████| 22500/22500 [00:00<00:00, 38046.73it/s]
/zfsstore/user/kreftb/QD-CNN/src/utilities/utils.py:386: RuntimeWarning: divide by zero encountered in divide
  return e/np.diag(C_DG)
Generated datapoint 00034 with 3 cuts
```


#### Example 4
The same error happens for the realizations (folder ID): 1613, 985
Error calls for few are printed below.

```
Generating realization 1613/5000...
EXPERIMENT INITIALIZED
-----------------------

                Capacitance model deployed with the following parameters:
                Dot-dot capacitances: 
                [[25.5707362   3.65126548  1.74036569  1.68730996]
 [ 3.65126548 22.69066761  3.70688734  0.68507701]
 [ 1.74036569  3.70688734 18.84715971  0.5255114 ]
 [ 1.68730996  0.68507701  0.5255114  13.66932637]]
                Dot-gate capacitances: 
                [[10.25310484  0.41498613  0.06630874  0.        ]
 [ 0.36387742 11.09315793  0.35189485  0.        ]
 [ 0.05851305  0.42631799  7.1564025   0.        ]
 [ 0.          0.          0.          5.        ]]
                Size of Coulomb peaks V[n] is constant
                

            Sensor model deployed with the following parameters:   
            Sensor dot indices: [3]
            Sensor detunings: [1000.] meV
            Coulomb peak width: 2.18 meV
            Slow noise amplitude: 0.0 ueV
            Fast noise amplitude: 0.0 ueV
            Signal noise scale: 0.0
            

            Tunneling model deployed with the following parameters:
            Tunneling matrix:
            [[0.00000000e+00 8.75963209e-06 2.34136422e-09 0.00000000e+00]
 [8.75963209e-06 0.00000000e+00 1.65192227e-05 0.00000000e+00]
 [2.34136422e-09 1.65192227e-05 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]]
            Temperature: 0.1 K
            Energy range factor: 3
            
Rastering CSD: 100%|██████████| 22500/22500 [00:00<00:00, 35442.01it/s]
Rastering CSD: 100%|██████████| 22500/22500 [00:00<00:00, 34648.70it/s]
Rastering CSD: 100%|██████████| 22500/22500 [00:00<00:00, 36739.69it/s]
Traceback (most recent call last):
  File "/zfsstore/user/kreftb/QD-CNN/src/utilities/utils.py", line 556, in generate_datapoint
    xout, yout, _, polytopes, sensor_values, _ = experiment.generate_CSD(
  File "/zfsstore/user/kreftb/QD-CNN/qdarts/experiment.py", line 522, in generate_CSD
    backend, CSD_data, states = get_CSD_data(
  File "/zfsstore/user/kreftb/QD-CNN/qdarts/plotting.py", line 141, in get_CSD_data
    corner_state = simulation.find_state_of_voltage(
  File "/zfsstore/user/kreftb/QD-CNN/qdarts/simulator.py", line 360, in find_state_of_voltage
    state, v_inside = self.find_boundary_intersection(
  File "/zfsstore/user/kreftb/QD-CNN/qdarts/simulator.py", line 324, in find_boundary_intersection
    transition_state = self.find_state_of_voltage(
  File "/zfsstore/user/kreftb/QD-CNN/qdarts/simulator.py", line 360, in find_state_of_voltage
    state, v_inside = self.find_boundary_intersection(
  File "/zfsstore/user/kreftb/QD-CNN/qdarts/simulator.py", line 322, in find_boundary_intersection
    raise LookupError()
LookupError
Generated datapoint 01612 with 3 cuts
```

```
EXPERIMENT INITIALIZED
-----------------------

                Capacitance model deployed with the following parameters:
                Dot-dot capacitances: 
                [[16.5423392   6.04392905  2.64560298  1.27983159]
 [ 6.04392905 22.75807129  1.65856587  1.06462403]
 [ 2.64560298  1.65856587 21.22489246  0.80376525]
 [ 1.27983159  1.06462403  0.80376525 15.93513273]]
                Dot-gate capacitances: 
                [[9.30558298 0.45733969 0.22157788 0.        ]
 [0.41120987 5.91717089 0.05541817 0.        ]
 [0.30575864 0.02250808 6.60672161 0.        ]
 [0.         0.         0.         5.        ]]
                Size of Coulomb peaks V[n] is constant
                

            Sensor model deployed with the following parameters:   
            Sensor dot indices: [3]
            Sensor detunings: [1000.] meV
            Coulomb peak width: 2.18 meV
            Slow noise amplitude: 0.0 ueV
            Fast noise amplitude: 0.0 ueV
            Signal noise scale: 0.0
            

            Tunneling model deployed with the following parameters:
            Tunneling matrix:
            [[0.00000000e+00 6.20397324e-05 5.90540664e-06 0.00000000e+00]
 [6.20397324e-05 0.00000000e+00 6.26961187e-09 0.00000000e+00]
 [5.90540664e-06 6.26961187e-09 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]]
            Temperature: 0.1 K
            Energy range factor: 3
            
Rastering CSD: 100%|██████████| 22500/22500 [00:00<00:00, 33744.38it/s]
Rastering CSD: 100%|██████████| 22500/22500 [00:00<00:00, 36919.10it/s]
Rastering CSD: 100%|██████████| 22500/22500 [00:00<00:00, 38157.02it/s]
Traceback (most recent call last):
  File "/zfsstore/user/kreftb/QD-CNN/src/utilities/utils.py", line 556, in generate_datapoint
    xout, yout, _, polytopes, sensor_values, _ = experiment.generate_CSD(
  File "/zfsstore/user/kreftb/QD-CNN/qdarts/experiment.py", line 522, in generate_CSD
    backend, CSD_data, states = get_CSD_data(
  File "/zfsstore/user/kreftb/QD-CNN/qdarts/plotting.py", line 141, in get_CSD_data
    corner_state = simulation.find_state_of_voltage(
  File "/zfsstore/user/kreftb/QD-CNN/qdarts/simulator.py", line 360, in find_state_of_voltage
    state, v_inside = self.find_boundary_intersection(
  File "/zfsstore/user/kreftb/QD-CNN/qdarts/simulator.py", line 324, in find_boundary_intersection
    transition_state = self.find_state_of_voltage(
  File "/zfsstore/user/kreftb/QD-CNN/qdarts/simulator.py", line 360, in find_state_of_voltage
    state, v_inside = self.find_boundary_intersection(
  File "/zfsstore/user/kreftb/QD-CNN/qdarts/simulator.py", line 322, in find_boundary_intersection
    raise LookupError()
LookupError
Generated datapoint 00984 with 3 cuts
Generating realization 986/5000...
```


#### Example 5
The same error happens for the realizations (folder ID): 1613, 
Error calls for few are printed below.
```
Generating realization 940/5000...
EXPERIMENT INITIALIZED
-----------------------

                Capacitance model deployed with the following parameters:
                Dot-dot capacitances: 
                [[32.10477113  4.51407422  2.6708979   1.82314199]
 [ 4.51407422 18.66446645  2.820647    1.44997632]
 [ 2.6708979   2.820647   15.07443923  0.817895  ]
 [ 1.82314199  1.44997632  0.817895   15.45889356]]
                Dot-gate capacitances: 
                [[11.35121492  0.29796048  0.66532095  0.        ]
 [ 0.06514935  2.43622712  0.04259562  0.        ]
 [ 0.21730586  0.10897175  6.5859053   0.        ]
 [ 0.          0.          0.          5.        ]]
                Size of Coulomb peaks V[n] is constant
                

            Sensor model deployed with the following parameters:   
            Sensor dot indices: [3]
            Sensor detunings: [1000.] meV
            Coulomb peak width: 2.18 meV
            Slow noise amplitude: 0.0 ueV
            Fast noise amplitude: 0.0 ueV
            Signal noise scale: 0.0
            

            Tunneling model deployed with the following parameters:
            Tunneling matrix:
            [[0.00000000e+00 3.41406485e-05 2.06544971e-05 0.00000000e+00]
 [3.41406485e-05 0.00000000e+00 2.89165826e-07 0.00000000e+00]
 [2.06544971e-05 2.89165826e-07 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]]
            Temperature: 0.1 K
            Energy range factor: 3
            
Rastering CSD: 100%|██████████| 22500/22500 [00:00<00:00, 35301.22it/s]
Rastering CSD: 100%|██████████| 22500/22500 [00:00<00:00, 44043.24it/s]
Rastering CSD: 100%|██████████| 22500/22500 [00:00<00:00, 39607.80it/s]
Traceback (most recent call last):
  File "/zfsstore/user/kreftb/QD-CNN/src/utilities/utils.py", line 365, in get_virtual_gate_transitions
    alpha_inv = np.linalg.inv(alpha)
  File "/zfsstore/user/kreftb/QD-CNN/env/lib64/python3.9/site-packages/numpy/linalg/_linalg.py", line 608, in inv
    ainv = _umath_linalg.inv(a, signature=signature)
  File "/zfsstore/user/kreftb/QD-CNN/env/lib64/python3.9/site-packages/numpy/linalg/_linalg.py", line 104, in _raise_linalgerror_singular
    raise LinAlgError("Singular matrix")
numpy.linalg.LinAlgError: Singular matrix

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/zfsstore/user/kreftb/QD-CNN/src/utilities/utils.py", line 525, in generate_datapoint
    transition_vectors = get_virtual_gate_transitions(
  File "/zfsstore/user/kreftb/QD-CNN/src/utilities/utils.py", line 367, in get_virtual_gate_transitions
    raise ValueError("Alpha matrix is singular; transitions are undefined.") from exc
ValueError: Alpha matrix is singular; transitions are undefined.
Generated datapoint 00939 with 3 cuts
Generating realization 941/5000...
```

TODO:
 - [v] Make sure that if there is an error, the instance will be completely erased, so that it will not have issues with ID number addressing empty or incomplete datapoints. In effect, the final ID number will be smaller than set at the begining.
 - [ ] Make GitHub issue out of them






