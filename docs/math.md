# Mathematical conventions

## State and indexing

An n-qubit pure state is a normalized vector of complex amplitudes:

$$
|\psi\rangle = \sum_{i=0}^{2^n-1} a_i |i\rangle,
\qquad \sum_i |a_i|^2 = 1.
$$

QuantaForge uses little-endian qubit indices: qubit `q` is bit `q` of integer `i`.
Printed kets put the most significant bit on the left, `|q[n-1] ... q[0]>`.
For two qubits the array order is `|00>, |01>, |10>, |11>`. Consequently X on
qubit 0 sends `|00>` to `|01>`; X on qubit 1 sends it to `|10>`.

CPU storage is NumPy `complex128`, consisting of double-precision real and
imaginary parts. The zero state has `a[0] = 1` and all other amplitudes zero.

## Single-qubit gates

For a target `t`, choose an index `i0` with bit `t` equal to zero and set
`i1 = i0 | (1 << t)`. A matrix U acts on that amplitude pair:

$$
\begin{pmatrix} a'_{i_0} \\ a'_{i_1} \end{pmatrix}
=
\begin{pmatrix} U_{00} & U_{01} \\ U_{10} & U_{11} \end{pmatrix}
\begin{pmatrix} a_{i_0} \\ a_{i_1} \end{pmatrix}.
$$

There are `2**(n-1)` independent pairs. Enumerating them avoids materializing a
full `2**n` by `2**n` operator. A future GPU kernel can map logical element `p` to a pair
by inserting a zero target bit:

```text
low  = p & ((1 << t) - 1)
high = p >> t
i0   = (high << (t + 1)) | low
i1   = i0 | (1 << t)
```

Every pair is disjoint, and both original amplitudes are needed before either
pair output can safely replace its input.

The implemented fixed gates use these matrices:

$$
X = \begin{pmatrix}0&1\\1&0\end{pmatrix},\quad
Y = \begin{pmatrix}0&-i\\i&0\end{pmatrix},\quad
Z = \begin{pmatrix}1&0\\0&-1\end{pmatrix},
$$

$$
H = \frac{1}{\sqrt{2}}\begin{pmatrix}1&1\\1&-1\end{pmatrix},\quad
S = \begin{pmatrix}1&0\\0&i\end{pmatrix},\quad
T = \begin{pmatrix}1&0\\0&e^{i\pi/4}\end{pmatrix}.
$$

In particular, `Y|0> = i|1>`; phase is part of the state and must not be discarded.
The rotation convention is `R_P(theta) = exp(-i theta P / 2)` for Pauli matrix P,
with angles in radians. Writing `c = cos(theta/2)` and `s = sin(theta/2)`:

$$
R_X(\theta)=\begin{pmatrix}c&-is\\-is&c\end{pmatrix},\quad
R_Y(\theta)=\begin{pmatrix}c&-s\\s&c\end{pmatrix},\quad
R_Z(\theta)=\begin{pmatrix}e^{-i\theta/2}&0\\0&e^{i\theta/2}\end{pmatrix}.
$$

`R_P(-theta)` is the inverse of `R_P(theta)`. H, X, Y, and Z are their own inverses.
All these matrices are unitary, so they preserve the norm in exact arithmetic.

## Controlled gates and circuit order

CX (CNOT) applies X to the target when the control bit is one; otherwise it leaves
the pair unchanged. CZ multiplies an amplitude by -1 exactly when both control
and target bits are one. Control and target must be distinct.

Operations execute in insertion order. If the circuit records U then V, its final
state is `V U |psi>`. For qubit ordering, a single-qubit matrix on qubit 0 of a
two-qubit register corresponds to `I ⊗ U`; on qubit 1 it corresponds to `U ⊗ I`.

Starting at `|00>`, H on qubit 0 then CX with control 0 and target 1 yields
the Bell state:

$$
\frac{|00\rangle + |11\rangle}{\sqrt{2}}.
$$

For an n-qubit GHZ state, apply H to qubit 0 and then CX from control 0 to every
other qubit. Only indices zero and `2**n - 1` have nonzero amplitude, both
`1/sqrt(2)`.

## Probabilities and sampling

The probability of basis outcome i is `p[i] = |a[i]|**2`. Sampling draws from
this categorical distribution with a local NumPy random generator. It returns
integer indices without collapsing the stored state; format an outcome with
`format(i, f"0{n}b")` to obtain the corresponding ket bitstring.

A global phase `exp(i phi)` changes no measurement probabilities. Exact circuit
semantics still specify output phases, so CPU gate tests compare amplitudes
directly rather than accepting arbitrary phase shifts that might hide a gate bug.

## Numerical checks and limits

Input states must be finite, with squared norm within an absolute `1e-12` of one.
The simulator does not renormalize after gates. Sampling rescales only its already
validated probability weights to accommodate floating-point drift in the RNG's
categorical distribution; it does not change the state or returned probabilities.
CPU tests use `rtol=0` and `atol=1e-12` for small `complex128`
states and circuits, with explicit seeds for random normalized inputs. Norm checks,
known phases, and dense independent small-system operators complement each other.
Probability checks alone cannot detect all phase or indexing errors.

Roundoff can accumulate with circuit depth. The CPU tolerance is a tested small-case
threshold, not a general error bound. Future split-`float32` GPU kernels require
device-backed error measurements and their own tolerances. The reference itself
also needs independent validation; agreement between two backends is insufficient
if both share the same mistake.
