import numpy as np
import pytest
from qiskit.quantum_info import Operator, Statevector
from qiskit.synthesis.qft import synth_qft_full

from quantum_subset_sum.subset_sum import *

# These tolerances are used since there should be no error in the tests below due to the low number of qubits.
ATOL = 1e-10
RTOL = 0


@pytest.mark.parametrize(
    "values, initial_state, expected_state",
    [
        # test cases are in the format (values, intial_sum_register+intial_index_register, expected_sum_register+expected_index_register)
        ([1, 1, 1, 1], "0000" + "1000", "0001" + "1000"),
        ([1, 1, 1, 1], "0000" + "1010", "0010" + "1010"),
        ([1, 2, 3, 1], "0000" + "1111", "0111" + "1111"),
        ([1, 2, 3], "0001" + "010", "0011" + "010"),
    ],
)
def test_controlled_draper_adder(values, initial_state, expected_state):

    # Arrange
    num_of_sum_qubits = int(np.ceil(np.log2(sum(values)) + 1))
    adder = controlled_draper_adder(values, num_of_sum_qubits)

    num_of_control_qubits = len(values)
    control_qubits = QuantumRegister(size=num_of_control_qubits, name="Control qubits")
    sum_register = QuantumRegister(size=num_of_sum_qubits, name="Sum Register")

    circuit = QuantumCircuit(control_qubits, sum_register)
    circuit.append(adder, control_qubits[:] + sum_register[:])

    state = Statevector.from_label(initial_state)

    # Act
    evolved_state = state.evolve(circuit)

    # Assert
    expected_state = Statevector.from_label(expected_state)
    np.testing.assert_allclose(evolved_state, expected_state, rtol=RTOL, atol=ATOL)


# This test is adapted from lab 2.
@pytest.mark.parametrize(
    "num_of_qubits",
    [
        (1),
        (2),
        (3),
        (4),
        (5),
    ],
)
def test_qft(num_of_qubits):

    # Arrange
    reg = QuantumRegister(num_of_qubits)
    qft_gate = QuantumCircuit(reg, name="QFT")
    qft_gate.append(qft(num_of_qubits), reg)

    # Act and Assert
    qiskit_qft_gate = synth_qft_full(num_of_qubits)

    np.testing.assert_allclose(
        Operator(qft_gate).data,
        Operator(qiskit_qft_gate).data,
        atol=ATOL,
        rtol=RTOL,
    )


# This test is adapted from lab 2.
@pytest.mark.parametrize(
    "num_of_qubits",
    [
        (1),
        (2),
        (3),
        (4),
        (5),
    ],
)
def test_iqft(num_of_qubits):

    # Arrange
    reg = QuantumRegister(num_of_qubits)
    iqft_gate = QuantumCircuit(reg, name="IQFT")
    iqft_gate.append(iqft(num_of_qubits), reg)

    # Act and Assert
    qiskit_iqft_gate = synth_qft_full(num_of_qubits, inverse=True)

    np.testing.assert_allclose(
        Operator(iqft_gate).data,
        Operator(qiskit_iqft_gate).data,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize(
    "bitstring, initial_state, expected_state",
    [
        ("0", Statevector.from_label("0"), Statevector.from_label("1")),
        ("0", Statevector.from_label("1"), Statevector.from_label("0")),
        ("1", Statevector.from_label("0"), Statevector.from_label("0")),
        ("1", Statevector.from_label("1"), Statevector.from_label("1")),
        ("010101", Statevector.from_label("000000"), Statevector.from_label("101010")),
        ("010101", Statevector.from_label("000111"), Statevector.from_label("101101")),
    ],
)
def test_flip_bits(bitstring, initial_state, expected_state):

    # Arrange
    sum_register = QuantumRegister(len(bitstring))

    circuit = QuantumCircuit(sum_register)
    circuit.append(
        flip_bits(bitstring),
        sum_register[:],
    )

    # Act
    evolved_state = initial_state.evolve(circuit)

    # Assert
    np.testing.assert_allclose(evolved_state, expected_state, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "target, sum_register_length, index_register_length, initial_state, expected_state",
    [
        (
            5,
            3,
            3,
            Statevector.from_label("101" + "000"),
            -Statevector.from_label("101" + "000"),
        ),
        (
            5,
            3,
            3,
            Statevector.from_label("010" + "000"),
            Statevector.from_label("010" + "000"),
        ),
        (
            15,
            4,
            4,
            Statevector.from_label("1111" + "1101"),
            -Statevector.from_label("1111" + "1101"),
        ),
        (
            15,
            4,
            4,
            Statevector.from_label("1101" + "1101"),
            Statevector.from_label("1101" + "1101"),
        ),
    ],
)
def test_phase_kickback(
    target, sum_register_length, index_register_length, initial_state, expected_state
):

    # Arrange
    sum_register = QuantumRegister(sum_register_length)
    index_register = QuantumRegister(index_register_length)

    circuit = QuantumCircuit(index_register, sum_register)
    circuit.append(
        phase_kickback(target, sum_register_length, index_register_length),
        index_register[:] + sum_register[:],
    )

    # Act
    evolved_state = initial_state.evolve(circuit)

    # Assert
    np.testing.assert_allclose(evolved_state, expected_state, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "input_list, target, initial_state, expected_state",
    [
        (
            [1, 1, 1],
            2,
            Statevector.from_label("000" + "101"),
            -Statevector.from_label("000" + "101"),
        ),
        (
            [1, 1, 1],
            2,
            Statevector.from_label("000" + "100"),
            Statevector.from_label("000" + "100"),
        ),
        (
            [1, 3, 5],
            8,
            Statevector.from_label("0000" + "110"),
            -Statevector.from_label("0000" + "110"),
        ),
        (
            [1, 3, 5],
            8,
            Statevector.from_label("0000" + "101"),
            Statevector.from_label("0000" + "101"),
        ),
        (
            [1, 1, 1, 1, 1],
            5,
            Statevector.from_label("000" + "10101"),
            Statevector.from_label("000" + "10101"),
        ),
        (
            [1, 1, 1, 1, 1],
            3,
            Statevector.from_label("000" + "10101"),
            -Statevector.from_label("000" + "10101"),
        ),
    ],
)
def test_phase_oracle(input_list, target, initial_state, expected_state):

    # Arrange
    n = len(input_list)
    total = sum(input_list)
    sum_register_length = int(np.ceil(np.log2(total + 1)))

    sum_register = QuantumRegister(size=sum_register_length, name="Sum Register")
    index_register = QuantumRegister(size=n, name="Index")

    circuit = QuantumCircuit(index_register, sum_register)
    circuit.append(
        phase_oracle(input_list, target), index_register[:] + sum_register[:]
    )

    # Act
    evolved_state = initial_state.evolve(circuit)

    # Assert
    np.testing.assert_allclose(evolved_state, expected_state, rtol=RTOL, atol=ATOL)
