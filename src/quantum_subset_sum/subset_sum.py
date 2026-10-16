import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.primitives import StatevectorSampler


def qft(num_of_qubits: int) -> Gate:
    """Constructs a QFT gate with n qubits. Assumed that n is positive. This code snippet is taken from lab 2.

    Args:
        num_of_qubits: Number of qubits to be used in the gate. Assumed to be a positive integer.

    Returns:
        A QFT gate with n qubits.
    """
    circuit = QuantumCircuit(num_of_qubits, name="QFT")
    for j in range(num_of_qubits // 2):
        circuit.swap(j, num_of_qubits - 1 - j)
    for j in range(num_of_qubits):
        circuit.h(j)
        for ell in range(j + 1, num_of_qubits):
            circuit.cp(np.pi / 2 ** (ell - j), j, ell)

    circuit = circuit.to_gate()
    circuit.label = "QFT"
    return circuit


def iqft(num_of_qubits: int) -> Gate:
    """Constructs an inverse QFT gate with n qubits. Assumed that n is positive. This code snippet is taken from lab 2.

    Args:
        num_of_qubits: Number of qubits to be used in the gate. Assumed to be a positive integer.

    Returns:
        An inverse QFT gate with n qubits.
    """
    circuit = QuantumCircuit(num_of_qubits, name="IQFT")
    for j in reversed(range(num_of_qubits)):
        for ell in reversed(range(j + 1, num_of_qubits)):
            circuit.cp(-np.pi / 2 ** (ell - j), j, ell)
        circuit.h(j)
    for j in reversed(range(num_of_qubits // 2)):
        circuit.swap(j, num_of_qubits - 1 - j)

    circuit = circuit.to_gate()
    circuit.label = "IQFT"
    return circuit


def controlled_draper_adder(values: list[int], num_of_sum_qubits: int) -> Gate:
    """Constructs a Draper adder that adds fixed classical integers to a sum register, conditionally on an index register.

    Args:
        values: The classical integers to be added to the sum register.
        num_of_sum_qubits: Size of the sum register.
    Returns:
        A controlled Draper's adder (with fixed constants) gate. The registers are ordered (the least significant bit first) with the index register first then the sum register.
    """
    num_of_control_qubits = len(values)
    index_register = QuantumRegister(size=num_of_control_qubits, name="Index Register")
    sum_register = QuantumRegister(size=num_of_sum_qubits, name="Sum Register")

    circuit = QuantumCircuit(index_register, sum_register)

    # Apply a qft to the sum register to transform it to the Fourier basis
    qft_circuit = qft(num_of_sum_qubits)
    circuit.append(qft_circuit, sum_register)

    # Add controlled phase gates to conditonally apply a phase change (i.e. addition)
    for i, val in enumerate(values):
        for j in range(num_of_sum_qubits):
            phase = 2 * np.pi * val * 2**j / 2**num_of_sum_qubits
            circuit.cp(phase, index_register[i], sum_register[j])

    # Apply an iqft to the sum register to transform back to the computational basis
    iqft_circuit = iqft(num_of_sum_qubits)
    circuit.append(iqft_circuit, sum_register)

    circuit = circuit.to_gate()
    circuit.label = "Controlled Draper QFT Adder"

    return circuit


def phase_kickback(
    target: int, sum_register_length: int, index_register_length: int
) -> Gate:
    """Constructs a gate that flips the phase of vectors that have the target sum.
    Args:
        target: The target sum. Assumed that target is positive and less that 2^sum_register_length.
        sum_register_length: Size of the sum register.
        index_register_length: Size of the index register.
    Returns:
        A phase kickback gate for the given target sum. The registers are ordered (the least significant bit first) with the index register first then the sum register.
    """
    target_bitstring = f"{target:b}".zfill(sum_register_length)

    sum_register = QuantumRegister(sum_register_length)
    index_register = QuantumRegister(index_register_length)

    circuit = QuantumCircuit(index_register, sum_register)

    # Flip the qubits of the sum register where the target bitstring is 0.
    # This means that the sum register is all ones exactly when the sum register is the target.
    flip_gate = flip_bits(target_bitstring)
    circuit.append(flip_gate, sum_register)

    # We want to flip the phase when the sum register bits are all 1.
    # Suppose the first qubit of the index register is 1, then a multicontrolled z gate (mcz gate) with target qubit on index register[0] does the job.
    # If it is 0 then a mcz gate keeps the phase the same. However conjugating the mcz gate with x gates flips the phase of the index.
    # Hence using an mcz gate and an mcz gate conjugated by x gates gives us exactly what we want.

    # Note that a mcz gate is the same as a mcx gate conjugated by Hadamard gates on the target register.

    # First mcz gate
    circuit.h(index_register[0])
    circuit.mcx(sum_register, index_register[0])
    circuit.h(index_register[0])

    # Second mcz gate conjugated by x gates
    circuit.x(index_register[0])
    circuit.h(index_register[0])
    circuit.mcx(sum_register, index_register[0])
    circuit.h(index_register[0])
    circuit.x(index_register[0])

    # Reverse the flip action from before. This is so we can reuse the ancilla sum register once we uncompute the sum.
    circuit.append(flip_gate, sum_register)

    circuit = circuit.to_gate()
    circuit.label = "Phase Kickback"

    return circuit


def flip_bits(bitstring: str):
    """A gate that applies an X gate to certain qubits.
    Args:
        bitstring: Bit string that controls which qubit indices have an X gate applied to.
    Returns:
        A gate that applies an X gate to qubits where the bit in the corresponding bitstring is 0. This is to be used on the sum register (the ancilla register).
    """

    register = QuantumRegister(len(bitstring))
    circuit = QuantumCircuit(register)

    for i, bit in enumerate(bitstring):
        # Apply an x gate if the corresponding bit in the bitstring is 0
        if bit == "0":
            # Note the first element in the register is the MSB, hence the reversal of the index below.
            circuit.x(register[len(bitstring) - 1 - i])

    circuit = circuit.to_gate()
    circuit.label = f"Flip qubits based on bitstring:{bitstring}"

    return circuit


def phase_oracle(input_list: list[int], target: int) -> Gate:
    """Constructs a phase oracle circuit for the subset sum problem.
    Args:
        input_list: The original set given. All the elements are assumed to be positive integers.
        target: The target sum. Assumed to be positive.
    Returns:
        A phase oracle gate for the subset sum problem for the given set and target. The registers are ordered (the least significant bit first) with the index register first then the sum register.
    """
    n = len(input_list)
    total = sum(input_list)
    sum_register_length = int(np.ceil(np.log2(total + 1)))

    sum_register = QuantumRegister(size=sum_register_length, name="Sum Register")
    index_register = QuantumRegister(size=n, name="Index")

    circuit = QuantumCircuit(index_register, sum_register)

    # Compute all subset sums.
    controlled_draper_adder_gate = controlled_draper_adder(
        input_list, sum_register_length
    )
    circuit.append(controlled_draper_adder_gate, index_register[:] + sum_register[:])

    # Invert the phase of the good subsets.
    phase_kickback_gate = phase_kickback(target, sum_register_length, n)
    circuit.append(phase_kickback_gate, index_register[:] + sum_register[:])

    # Uncompute the sum. This is the same as adding by the negative values.
    negative_input_list = [-val for val in input_list]
    inverse_controlled_draper_adder_gate = controlled_draper_adder(
        negative_input_list, sum_register_length
    )
    circuit.append(
        inverse_controlled_draper_adder_gate, index_register[:] + sum_register[:]
    )

    circuit = circuit.to_gate()
    circuit.label = "Phase Oracle"
    return circuit


def grover_diffuser(num_of_qubits: int) -> Gate:
    """Constructs a Grover diffuser gate for n qubits.
    Args:
        num_of_qubits: Number of qubits. Assumed to be positive.
    Returns:
        A Grover diffuser gate for n qubits (to be used on the index register).
    """
    index_register = QuantumRegister(size=num_of_qubits, name="Index")

    circuit = QuantumCircuit(index_register)
    circuit.h(index_register)
    circuit.x(index_register)

    circuit.h(index_register[-1])
    circuit.mcx(index_register[:-1], index_register[-1])
    circuit.h(index_register[-1])

    circuit.x(index_register)
    circuit.h(index_register)

    circuit = circuit.to_gate()
    circuit.label = "Diffuser"

    return circuit


def grover_iteration(input_list: list[int], target: int) -> Gate:
    """
    Args:
        input_list: The original set given. All the elements are assumed to be positive integers.
        target: The target sum. Assumed to be positive.
    Returns:
        A Grover iteration gate. The registers are ordered (the least significant bit first) with the index register first then the sum register.
    """
    n = len(input_list)
    total = sum(input_list)

    sum_register_length = int(np.ceil(np.log2(total + 1)))

    sum_register = QuantumRegister(size=sum_register_length, name="Sum Register")
    index_register = QuantumRegister(size=n, name="Index")

    circuit = QuantumCircuit(index_register, sum_register)

    # An iteration is simply the oracle followed by the diffuser.
    phase_oracle_gate = phase_oracle(input_list, target)
    grover_diffuser_gate = grover_diffuser(n)

    circuit.append(phase_oracle_gate, index_register[:] + sum_register[:])
    circuit.append(grover_diffuser_gate, index_register[:])

    circuit = circuit.to_gate()
    circuit.label = "Iteration"

    return circuit


def build_subset_sum_circuit(
    input_list: list[int], target: int, number_of_iterations: int
):
    """Constructs an entire quantum circuit to solve the subset sum problem for a fixed set and target.
    Args:
        input_list: The original set given. All the elements are assumed to be positive integers.
        target: The target sum. Assumed to be positive.
        number_of_iterations: The number of Grover iterations to be used in the circuit.
    Returns:
        A quantum circuit to solve the subset sum problem for a fixed set and target. The registers are ordered (the least significant bit first) with the index register first then the sum register.
    """
    n = len(input_list)
    total = sum(input_list)

    sum_register_length = int(np.ceil(np.log2(total + 1)))

    sum_register = QuantumRegister(size=sum_register_length, name="Sum Register")
    index_register = QuantumRegister(size=n, name="Index")

    quantum_circuit = QuantumCircuit(index_register, sum_register)

    # Set the index qubits to be in equal superposition. This models all possible subsets of our original list.
    quantum_circuit.h(index_register)

    # Add the specified number of Grover iterations to the circuit.
    iteration_gate = grover_iteration(input_list, target)
    for _ in range(number_of_iterations):
        quantum_circuit.append(iteration_gate, index_register[:] + sum_register[:])

    # Two classical registers are added for measurement at the end.
    classical_output_register = ClassicalRegister(n, name="Measurement")
    ancilla_register_check = ClassicalRegister(sum_register_length, name="check")

    quantum_circuit.add_register(classical_output_register)
    quantum_circuit.add_register(ancilla_register_check)
    quantum_circuit.measure(index_register, classical_output_register)
    quantum_circuit.measure(sum_register, ancilla_register_check)
    return quantum_circuit


def compute_optimal_iterations(number_of_elements: int, number_of_solutions: int):
    """Computes the optimal number of Grover iterations.
    Args:
        number_of_elements: The size of the set.
        number_of_solutions: The number of assumed solutions to our subset problem.
    Returns:
        The closest integer to np.pi / (4 * theta) - 0.5,
        where theta = arcsin(sqrt(number_of_solutions / 2^number_of_elements)
    """
    theta = np.arcsin(np.sqrt(number_of_solutions / 2**number_of_elements))
    number_of_iterations = int(np.rint(np.pi / (4 * theta) - 0.5))
    return number_of_iterations


def sampling_routine(input_list: list[int], target: int):
    """A sampling routine to solve the subset sum problem when the number of solutions is not known in advance.
    Args:
        input_list: The original set given. All the elements are assumed to be positive integers.
        target: The target sum. Assumed to be positive.
    Returns:
        A sampling routine to solve the subset sum problem.
    """
    n = len(input_list)

    subset = []

    if sum(input_list) < target:
        print(
            f"Maximum possible sum for the input list, {sum(input_list)}, is less than the target, {target}"
        )
        return

    for i in range(n):
        number_of_iterations = compute_optimal_iterations(n, 2**i)
        circuit = build_subset_sum_circuit(input_list, target, number_of_iterations)

        sampler = StatevectorSampler()
        result = sampler.run([circuit], shots=1).result()
        dist = result[0].data.Measurement.get_counts()
        indicies = next(iter(dist.keys()))[::-1]
        subset = [val for i, val in enumerate(input_list) if indicies[i] == "1"]

        if sum(subset) == target:
            print(f"Found subset: {subset}, with target sum {target}")
            return

    print(f"No subset found, with target sum {target}, within budget")
