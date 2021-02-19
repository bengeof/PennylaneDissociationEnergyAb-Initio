import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np



geometry = 'o2.xyz'



charge = 0


multiplicity = 1


basis_set = 'sto-3g'


name = 'o2'



h, qubits = qchem.molecular_hamiltonian(
    name,
    geometry,
    charge=charge,
    mult=multiplicity,
    basis=basis_set,
    active_electrons=8,
    active_orbitals=6,
    mapping='jordan_wigner'
)

print('Number of qubits = ', qubits)
print('Hamiltonian is ', h)



dev = qml.device('default.qubit', wires=qubits)


def circuit(params, wires):
    qml.BasisState(np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], requires_grad=False), wires=wires)
    for i in wires:
        qml.Rot(*params[i], wires=i)
    qml.CNOT(wires=[5, 11])
    qml.CNOT(wires=[4, 10])
    qml.CNOT(wires=[3, 9])
    qml.CNOT(wires=[2, 8])
    qml.CNOT(wires=[1, 7])
    qml.CNOT(wires=[0, 6])
    




cost_fn = qml.ExpvalCost(circuit, h, dev)




opt = qml.GradientDescentOptimizer(stepsize=0.4)
np.random.seed(0)
params = np.random.normal(0, np.pi, (qubits, 3))

print(params)



max_iterations = 200
conv_tol = 1e-06


for n in range(max_iterations):
    params, prev_energy = opt.step_and_cost(cost_fn, params)
    energy = cost_fn(params)
    conv = np.abs(energy - prev_energy)

    if n % 20 == 0:
        print('Iteration = {:},  Energy = {:.8f} Ha'.format(n, energy))

    if conv <= conv_tol:
        break

print()
print('Final convergence parameter = {:.8f} Ha'.format(conv))
print('Final value of the ground-state energy = {:.8f} Ha'.format(energy))
print('Accuracy with respect to the FCI energy: {:.8f} Ha ({:.8f} kcal/mol)'.format(
    np.abs(energy - (-1.136189454088)), np.abs(energy - (-1.136189454088))*627.503
    )
)
print()
print('Final circuit parameters = \n', params)


