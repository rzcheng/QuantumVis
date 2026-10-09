import { useState } from 'react';
import { assetUrl } from '../data';
import type { CircuitData, Operation } from '../types';

function angleLabel(index: number): string {
  if (index === 0) return '0';
  if (index === 32) return '2π';
  const divisor = [16, 8, 4, 2, 1].find(value => index % value === 0)!;
  const numerator = index / divisor;
  const denominator = 16 / divisor;
  return `${numerator === 1 ? '' : numerator}π${denominator === 1 ? '' : `/${denominator}`}`;
}

function operationLabel(gate: Operation) {
  return gate.control === null ? `${gate.name} on q${gate.target}` : `CX q${gate.control} → q${gate.target}`;
}

function CircuitDiagram({ qubits, operations, step, onStep }: {
  qubits: number; operations: Operation[]; step: number; onStep: (step: number) => void;
}) {
  const width = 360;
  const y = (q: number) => 34 + q * 54;
  return <svg className="circuit-diagram" viewBox={`0 0 ${width} ${qubits * 54 + 38}`}
    aria-label="Circuit gates. Select a gate to view the state after it.">
    {Array.from({ length: qubits }, (_, q) => <g key={q}>
      <text x="4" y={y(q) + 5} className="mono">q{q}</text>
      <line className="wire" x1="39" x2="350" y1={y(q)} y2={y(q)} />
    </g>)}
    {operations.map((gate, index) => {
      const x = 88 + index * 100;
      const selected = step === index + 1;
      return <g key={index} role="button" tabIndex={0} aria-pressed={selected}
        aria-label={`Step ${index + 1}: ${operationLabel(gate)}`}
        className={`gate ${selected ? 'selected' : index < step ? 'completed' : 'upcoming'}`}
        onClick={() => onStep(index + 1)} onKeyDown={event => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault(); onStep(index + 1);
          }
        }}>
        <rect className="gate-hit" x={x - 29} y="7" width="58" height={qubits * 54 - 1} rx="4" />
        {gate.control !== null ? <>
          <line className="gate-line" x1={x} x2={x} y1={y(gate.control)} y2={y(gate.target)} />
          <circle className="control-dot" cx={x} cy={y(gate.control)} r="5" />
          <circle className="gate-box" cx={x} cy={y(gate.target)} r="17" />
          <path className="gate-line" d={`M${x - 10},${y(gate.target)}h20 M${x},${y(gate.target) - 10}v20`} />
        </> : <>
          <rect className="gate-box" x={x - 23} y={y(gate.target) - 20} width="46" height="40" rx="4" />
          <text textAnchor="middle" x={x} y={y(gate.target) + 5}>{gate.name}</text>
        </>}
        <text className="step-number" textAnchor="middle" x={x} y={qubits * 54 + 25}>
          {selected ? `▴ ${index + 1}` : index + 1}
        </text>
      </g>;
    })}
  </svg>;
}

function PhaseDial({ real, imag }: { real: number; imag: number }) {
  if (Math.hypot(real, imag) < 1e-12) return <span className="phase-zero" aria-label="Phase undefined at zero amplitude">—</span>;
  const phase = Math.atan2(imag, real);
  return <svg className="phase-dial" viewBox="0 0 36 36" role="img"
    aria-label={`Phase ${(phase * 180 / Math.PI).toFixed(1)} degrees`}>
    <circle cx="18" cy="18" r="14" />
    <path className="phase-axis" d="M18 3v30 M3 18h30" />
    <line x1="18" y1="18" x2={18 + 13 * Math.cos(phase)} y2={18 - 13 * Math.sin(phase)} />
  </svg>;
}

function amplitudeLabel(real: number, imag: number) {
  const clean = (value: number) => (Math.abs(value) < 0.00005 ? 0 : value);
  return `${clean(real).toFixed(4)} ${clean(imag) < 0 ? '−' : '+'} ${Math.abs(clean(imag)).toFixed(4)}i`;
}

export default function CircuitExplorer({ data }: { data: CircuitData }) {
  const [preset, setPreset] = useState(0);
  const [step, setStep] = useState(2);
  const [angle, setAngle] = useState(8);
  const circuit = data.circuits[preset];
  const interference = circuit.id === 'interference';
  const recording = circuit.recordings[interference ? angle : 0];
  const last = recording.operations.length;
  const snapshot = recording.steps[step];

  return <>
    <div className="control-line">
      <div className="labeled-control"><label htmlFor="preset">Preset</label><select id="preset" value={preset} onChange={event => {
        const selected = Number(event.target.value);
        setPreset(selected); setStep(data.circuits[selected].recordings[0].operations.length);
      }}>{data.circuits.map((circuit, index) => <option key={circuit.id} value={index}>
        {circuit.name} · {circuit.num_qubits} {circuit.num_qubits === 1 ? 'qubit' : 'qubits'}
      </option>)}</select></div>
      <p className="provenance">Precomputed with QuantumVis · NumPy CPU · complex128</p>
    </div>
    {interference && <div className="angle-control">
      <label htmlFor="recorded-angle">recorded angle</label>
      <output htmlFor="recorded-angle" className="mono">{angleLabel(angle)} · {recording.angle_radians!.toFixed(4)} rad</output>
      <input id="recorded-angle" type="range" min="0" max="32" step="1" value={angle}
        aria-valuetext={`${angleLabel(angle)} radians, recorded setting ${angle + 1} of 33`}
        onChange={event => setAngle(Number(event.target.value))} />
      <div className="range-labels"><span>0</span><span>π</span><span>2π</span></div>
      <p>33 recorded settings, spaced π/16 apart. Each selects an exported snapshot; quantum states are never interpolated.</p>
    </div>}
    <div className="surface circuit-surface">
      <CircuitDiagram qubits={circuit.num_qubits} operations={recording.operations} step={step} onStep={setStep} />
      <div className="step-controls">
        <div className="button-group">
          <button disabled={step === 0} onClick={() => setStep(0)}>Initial</button>
          <button disabled={step === 0} onClick={() => setStep(step - 1)}>Previous</button>
          <button disabled={step === last} onClick={() => setStep(step + 1)}>Next</button>
        </div>
        <p role="status">step {step} of {last} <span className="muted">· {step === 0 ? 'Initial state' : operationLabel(recording.operations[step - 1])}</span></p>
      </div>
    </div>
    <p className="bit-order">q0 is the least significant bit. Basis labels read q[n−1]…q0.</p>
    <div className="state-table" role="table" aria-label="Recorded state vector">
      <div className="state-header" role="row">
        <span role="columnheader">Basis</span><span role="columnheader">Probability · 0–100%</span>
        <span role="columnheader">Complex amplitude</span><span role="columnheader">Phase</span>
      </div>
      {snapshot.real.map((real, index) => <div className="state-row" role="row" key={index}>
        <span className="basis mono" role="cell">|{index.toString(2).padStart(circuit.num_qubits, '0')}⟩</span>
        <div className="probability" role="cell">
          <div className="bar-track" aria-hidden="true"><div className="bar" style={{ width: `${snapshot.probabilities[index] * 100}%` }} /></div>
          <span className="mono">{(snapshot.probabilities[index] * 100).toFixed(1)}%</span>
        </div>
        <span className="amplitude mono" role="cell"><span className="mobile-label">Amplitude </span>{amplitudeLabel(real, snapshot.imag[index])}</span>
        <span className="phase" role="cell"><PhaseDial real={real} imag={snapshot.imag[index]} /></span>
      </div>)}
    </div>
    {interference ? <p>RZ changes the relative phase while leaving probabilities unchanged at that step. The final H reveals interference.</p>
      : <p>Bell and GHZ recordings use the CPU backend. Controlled GPU gates are not implemented.</p>}
    <details>
      <summary>Reading phase and the recorded data</summary>
      <p>The dial starts at the right for phase 0 and turns counterclockwise. Relative phase compares amplitudes and affects how they interfere. An overall global phase does not change measurement probabilities.</p>
      <p>Zero amplitude has undefined phase (—). For display, magnitudes below 10⁻¹² also show a dash; amplitudes are rounded to four decimals. The exported values retain full numerical precision.</p>
      <p>Browser interactions replay exported results. Python, Triton, CUDA, and quantum simulation do not run in this browser.</p>
      <p>Export: Python {data.metadata.python_version} · NumPy {data.metadata.numpy_version}. Every initial state and gate prefix was run independently with the CPU backend. Source SHA-256 hashes and bit order are in the <a href={assetUrl('circuits.json')} download>circuit JSON</a>.</p>
    </details>
  </>;
}
