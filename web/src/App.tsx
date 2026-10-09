import { useEffect, useState } from 'react';
import BenchmarkPlot from './components/BenchmarkPlot';
import CircuitExplorer from './components/CircuitExplorer';
import KernelMapping from './components/KernelMapping';
import { fetchRecorded, parseBenchmark, parseCircuits, REPOSITORY, SOURCE } from './data';
import type { BenchmarkData, CircuitData } from './types';

function useRecorded<T>(filename: string, parse: (value: unknown) => T) {
  const [data, setData] = useState<T>();
  const [error, setError] = useState('');
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setData(undefined); setError('');
    fetchRecorded(filename, parse, controller.signal).then(setData).catch((error: unknown) => {
      if (!controller.signal.aborted) setError(error instanceof Error ? error.message : 'Could not load recorded data.');
    });
    return () => controller.abort();
  }, [filename, parse, attempt]);
  return { data, error, retry: () => setAttempt(attempt + 1) };
}

function DataMessage({ error, retry }: { error: string; retry: () => void }) {
  return error ? <div className="data-error" role="alert"><p>Recorded data unavailable. {error}</p>
    <button onClick={retry}>Retry loading</button></div> : <p role="status">Loading recorded data…</p>;
}

export default function App() {
  const circuits = useRecorded<CircuitData>('circuits.json', parseCircuits);
  const benchmark = useRecorded<BenchmarkData>('cpu-baseline.json', parseBenchmark);
  return <div className="page">
    <a className="skip-link" href="#main">Skip to content</a>
    <header>
      <a className="wordmark" href="#main">QuantumVis</a>
      <nav aria-label="Page sections">
        <a href="#circuit">Circuit</a><a href="#kernel">Kernel</a><a href="#benchmarks">Benchmarks</a>
        <a href={REPOSITORY}>GitHub ↗</a>
      </nav>
    </header>
    <main id="main" tabIndex={-1}>
      <div className="intro">
        <h1>Quantum circuit simulation,<br className="desktop-break" /> from state vectors to GPU kernels.</h1>
        <p>A NumPy reference for quantum state-vector simulation. The Triton work maps single-qubit gates to disjoint amplitude pairs.</p>
        <p className="status-line">CPU reference implemented · NVIDIA device measurements pending</p>
        <p className="browser-note">An interactive view of exported results. All circuit states were precomputed on the CPU.</p>
      </div>
      <section id="circuit" aria-labelledby="circuit-title">
        <div className="section-heading"><span className="section-number">01</span><h2 id="circuit-title">Circuit explorer</h2></div>
        {circuits.data ? <CircuitExplorer data={circuits.data} /> : <DataMessage {...circuits} />}
      </section>
      <section id="kernel" aria-labelledby="kernel-title">
        <div className="section-heading"><span className="section-number">02</span><h2 id="kernel-title">Inside the kernel</h2></div>
        <KernelMapping />
      </section>
      <section id="benchmarks" aria-labelledby="benchmarks-title">
        <div className="section-heading"><span className="section-number">03</span><h2 id="benchmarks-title">Recorded CPU benchmarks</h2></div>
        {benchmark.data ? <BenchmarkPlot data={benchmark.data} /> : <DataMessage {...benchmark} />}
      </section>
    </main>
    <footer><span>Ryan Cheng · Cornell ECE</span><div><a href={REPOSITORY}>Repository</a><a href={`${SOURCE}/docs/architecture.md`}>Technical documentation</a></div></footer>
  </div>;
}
