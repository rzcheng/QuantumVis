import { useEffect, useState } from 'react';
import { assetUrl, SOURCE } from '../data';
import type { BenchmarkCase, BenchmarkData } from '../types';

export const milliseconds = (nanoseconds: number) => nanoseconds / 1_000_000;

function Plot({ cases }: { cases: BenchmarkCase[] }) {
  const [compact, setCompact] = useState(() => window.matchMedia('(max-width: 650px)').matches);
  useEffect(() => {
    const query = window.matchMedia('(max-width: 650px)');
    const update = () => setCompact(query.matches);
    query.addEventListener('change', update);
    return () => query.removeEventListener('change', update);
  }, []);
  const left = compact ? 48 : 66;
  const right = compact ? 338 : 620;
  const values = cases.flatMap(entry => [milliseconds(entry.median_ns), milliseconds(entry.p95_ns)]);
  const low = Math.floor(Math.log10(Math.min(...values)));
  const high = Math.ceil(Math.log10(Math.max(...values)));
  const ticks = Array.from({ length: high - low + 1 }, (_, i) => 10 ** (low + i));
  const x = (qubits: number) => left + (qubits - 8) / 10 * (right - left);
  const y = (ms: number) => 250 - (Math.log10(ms) - low) / (high - low) * 212;
  return <svg className="benchmark-plot" viewBox={`0 0 ${compact ? 360 : 650} 308`} role="img"
    aria-label="Recorded CPU latency in milliseconds on a logarithmic y-axis. Median lines and 95th percentile markers; values in the table below.">
    <text x={left} y="19" className="axis-title">Latency (ms) · logarithmic scale</text>
    {ticks.map(tick => <g key={tick}>
      <line className="grid-line" x1={left} x2={right} y1={y(tick)} y2={y(tick)} />
      <text className="tick" x={left - 12} y={y(tick) + 5} textAnchor="end">{tick}</text>
    </g>)}
    {[8, 12, 16, 18].map(n => <g key={n}>
      <text className="tick" x={x(n)} y="275" textAnchor="middle">{n}</text>
      <line className="axis-mark" x1={x(n)} x2={x(n)} y1="250" y2="256" />
    </g>)}
    <text className="axis-title" x={(left + right) / 2} y="302" textAnchor="middle">Qubits</text>
    {[false, true].map(highTarget => {
      const points = cases.filter(entry => highTarget ? entry.target !== 0 : entry.target === 0)
        .sort((a, b) => a.num_qubits - b.num_qubits);
      return <g className={highTarget ? 'high-series' : 'low-series'} key={String(highTarget)}>
        <polyline className="median-line" strokeDasharray={highTarget ? '7 5' : undefined}
          points={points.map(entry => `${x(entry.num_qubits)},${y(milliseconds(entry.median_ns))}`).join(' ')} />
        {points.map(entry => {
          const px = x(entry.num_qubits);
          const medianY = y(milliseconds(entry.median_ns));
          const p95Y = y(milliseconds(entry.p95_ns));
          return <g key={entry.num_qubits}>
            {highTarget ? <rect className="median-marker" x={px - 4} y={medianY - 4} width="8" height="8" />
              : <circle className="median-marker" cx={px} cy={medianY} r="4" />}
            <path className="percentile-marker" d={`M${px - 5},${p95Y - 5}l10,10 m-10,0l10,-10`} />
          </g>;
        })}
      </g>;
    })}
  </svg>;
}

export default function BenchmarkPlot({ data }: { data: BenchmarkData }) {
  const [operation, setOperation] = useState<'H' | 'RX'>('H');
  const cases = data.cases.filter(entry => entry.operation === operation);
  const metadata = data.metadata;
  return <>
    <div className="control-line">
      <div className="labeled-control"><label htmlFor="benchmark-gate">Gate</label><select id="benchmark-gate" value={operation} onChange={event => setOperation(event.target.value as 'H' | 'RX')}>
        <option value="H">H</option><option value="RX">RX</option>
      </select></div>
      <p className="provenance">Recorded CPU baseline · Apple M4 Pro · complex128</p>
    </div>
    <p className="section-note">Complete single-gate run() latency; includes validation and copies</p>
    <div className="surface benchmark-surface">
      <div className="plot-legend">
        <span className="low-legend">● ━ Target 0 median</span>
        <span className="high-legend">■ ┄ Target n−1 median</span>
        <span>× p95 (95th percentile)</span>
      </div>
      <Plot cases={cases} />
    </div>
    <p className="muted">Five warmups and 31 trials per case. p95 is a percentile of the recorded trials, not a confidence interval. Lines connect measured sizes only.</p>
    <table className="benchmark-table">
      <caption>{operation} recorded latency in ms{operation === 'RX' ? ` · angle ${cases[0].angle_radians} rad` : ''}</caption>
      <thead><tr><th scope="col">Qubits</th><th scope="col">Target</th><th scope="col">Median (ms)</th><th scope="col">p95 (ms)</th></tr></thead>
      <tbody>{[...cases].sort((a, b) => a.num_qubits - b.num_qubits || a.target - b.target).map(entry =>
        <tr key={`${entry.num_qubits}-${entry.target}`}>
          <th scope="row">{entry.num_qubits}</th><td>{entry.target === 0 ? '0' : `${entry.target} (n−1)`}</td>
          <td>{milliseconds(entry.median_ns).toFixed(6)}</td><td>{milliseconds(entry.p95_ns).toFixed(6)}</td>
        </tr>)}</tbody>
    </table>
    <details className="benchmark-details">
      <summary>Recording details & raw data</summary>
      <dl>
        <dt>Recorded</dt><dd>{metadata.timestamp_utc}</dd>
        <dt>Environment</dt><dd>{metadata.cpu_model} · {metadata.platform}</dd>
        <dt>Versions</dt><dd>Python {metadata.python_version} · NumPy {metadata.numpy_version} · historical package {metadata.quantaforge_version}</dd>
        <dt>Method</dt><dd>Seed {data.seed} · 5 warmups · 31 trials per case · percentile method: {data.percentile_method}</dd>
        <dt>Source revision</dt><dd className="mono">{metadata.git_revision}</dd>
        <dt>Working tree</dt><dd>{metadata.git_dirty ? 'Dirty at recording time' : 'Clean at recording time'}. The recorded source hashes identify the measured files.</dd>
      </dl>
      <p>The old project name in the raw metadata is historical. The original artifact is copied byte-for-byte; measurements and provenance are unchanged.</p>
      <p><a href={assetUrl('cpu-baseline.json')} download>Download raw benchmark JSON</a> · <a href={`${SOURCE}/docs/benchmarking.md`}>Methodology ↗</a></p>
      <details>
        <summary>Recorded source SHA-256 hashes</summary>
        <ul className="source-hashes">{Object.entries(metadata.source_sha256).map(([path, hash]) =>
          <li key={path}><code>{path}</code><br /><code>{hash}</code></li>)}</ul>
      </details>
    </details>
  </>;
}
