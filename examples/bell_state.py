"""prepare (|00> + |11>)/sqrt(2) and verify its amplitudes."""

import numpy as np

from quantumvis import Circuit, CPUSimulator


def main() -> None:
    circuit = Circuit(2).h(0).cx(0, 1)
    result = CPUSimulator().run(circuit)
    expected = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
    np.testing.assert_allclose(result.amplitudes, expected, rtol=1e-12, atol=1e-12)

    print("Bell state: (|00> + |11>)/sqrt(2)")
    for index, probability in enumerate(result.probabilities()):
        print(f"  |{index:02b}>: {probability:.6f}")
    outcomes = result.sample(16, seed=42)
    print("16 seeded samples:", " ".join(f"{index:02b}" for index in outcomes))


if __name__ == "__main__":
    main()
