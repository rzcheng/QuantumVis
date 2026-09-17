import { useState } from 'react';
import { SOURCE } from '../data';

export function indexPairs(target: number) {
  return Array.from({ length: 4 }, (_, pair) => {
    const lowMask = (1 << target) - 1;
    const i0 = ((pair >> target) << (target + 1)) | (pair & lowMask);
    return [i0, i0 ^ (1 << target)] as const;
  });
}

export default function KernelMapping() {
  const [target, setTarget] = useState(0);
  const [pair, setPair] = useState(0);
  const pairs = indexPairs(target);
  const [i0, i1] = pairs[pair];
  const binary = (index: number) => index.toString(2).padStart(3, '0');

  return <>
    <p className="section-note">Indexing illustration of the Triton implementation</p>
    <div className="control-line">
      <div className="labeled-control"><label htmlFor="target">Target</label><select id="target" value={target} onChange={event => {
        setTarget(Number(event.target.value)); setPair(0);
      }}>{[0, 1, 2].map(q => <option key={q} value={q}>q{q}</option>)}</select></div>
      <span className="muted">3 qubits · 8 amplitudes · 4 disjoint pairs</span>
    </div>
    <fieldset className="pair-selector">
      <legend>Amplitude-index pair</legend>
      {pairs.map(([low, high], index) => <label className={pair === index ? 'active' : ''} key={index}>
        <input type="radio" name="pair" checked={pair === index} onChange={() => setPair(index)} />
        <span>Pair {index}<strong className="mono">{low} ↔ {high}</strong></span>
      </label>)}
    </fieldset>
    <div className="surface mapping-surface">
      <div className="storage" aria-label="Split real and imaginary storage">
        {['real', 'imaginary'].map(row => <div className="storage-row" key={row}>
          <span className="storage-label">{row}</span>
          {Array.from({ length: 8 }, (_, index) => <div key={index}
            className={`cell mono ${index === i0 || index === i1 ? 'selected-cell' : ''}`}
            aria-label={`${row} index ${index}${index === i0 || index === i1 ? ', selected' : ''}`}>
            {index}<span>{index === i0 ? 'i0' : index === i1 ? 'i1' : '·'}</span>
          </div>)}
        </div>)}
      </div>
      <div className="mapping-detail">
        <div aria-live="polite">
          <p className="pair-heading">Pair {pair}: <span className="mono">{i0} ↔ {i1}</span></p>
          <p className="mono binary-labels">|{binary(i0)}⟩ ↔ |{binary(i1)}⟩</p>
          <p>Only bit q{target} differs.</p>
          <code>i1 = i0 XOR (1 &lt;&lt; target)</code>
          <p className="mono">{i1} = {i0} XOR {1 << target}</p>
        </div>
        <div className="transformation">
          <p>One 2×2 gate, two outputs</p>
          <svg viewBox="0 0 300 88" role="img" aria-label="b0 equals u00 a0 plus u01 a1; b1 equals u10 a0 plus u11 a1">
            <g className="matrix-brackets"><path d="M25 12H18V76H25 M59 12H66V76H59 M106 12H99V76H106 M207 12H214V76H207 M239 12H232V76H239 M282 12H289V76H282" /></g>
            <g className="mono matrix-text" textAnchor="middle">
              <text x="42" y="36">b₀</text><text x="42" y="64">b₁</text><text x="83" y="50">=</text>
              <text x="127" y="36">u₀₀</text><text x="184" y="36">u₀₁</text>
              <text x="127" y="64">u₁₀</text><text x="184" y="64">u₁₁</text>
              <text x="260" y="36">a₀</text><text x="260" y="64">a₁</text>
            </g>
          </svg>
          <p className="muted">a₀ = ψ[{i0}] · a₁ = ψ[{i1}]<br />b₀ and b₁ write back to the same indices.</p>
        </div>
      </div>
    </div>
    <p>Each logical Triton element loads both complex amplitudes before writing the two outputs. Every pair has one owner, so pairs do not overlap.</p>
    <p className="muted">This illustrates index arithmetic, not a device trace, measured bandwidth, or physical GPU threads. A logical element does not imply one CUDA thread. The implementation uses separate float32 real and imaginary arrays.</p>
    <a href={`${SOURCE}/src/quantumvis/gpu/kernels/single_qubit.py`}>Read single_qubit.py ↗</a>
  </>;
}
