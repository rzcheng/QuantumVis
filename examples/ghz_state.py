"""prepare a four-qubit ghz state and verify its amplitudes."""

import numpy as np

from quantumvis import Circuit, CPUSimulator


def main() -> None:
    num_qubits = 4
    circuit = Circuit(num_qubits).h(0)
    for target in range(1, num_qubits):
        circuit.cx(0, target)

    result = CPUSimulator().run(circuit)
    expected = np.zeros(2**num_qubits, dtype=np.complex128)
    expected[[0, -1]] = 1 / np.sqrt(2)
    np.testing.assert_allclose(result.amplitudes, expected, rtol=1e-12, atol=1e-12)

    print(f"{num_qubits}-qubit GHZ state: (|0000> + |1111>)/sqrt(2)")
    print("Nonzero computational-basis probabilities:")
    for index, probability in enumerate(result.probabilities()):
        if probability > 0:
            print(f"  |{index:0{num_qubits}b}>: {probability:.6f}")
    outcomes = result.sample(16, seed=42)
    print("16 seeded samples:", " ".join(f"{index:0{num_qubits}b}" for index in outcomes))


if __name__ == "__main__":
    main()
