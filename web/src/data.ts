import type { BenchmarkData, CircuitData } from './types';

export const REPOSITORY = 'https://github.com/rzcheng/QuantumVis';
export const SOURCE = `${REPOSITORY}/blob/3fc8db12c7eafd3f08db1e12405b5aff356dbe2f`;
export const assetUrl = (name: string) => `${import.meta.env.BASE_URL}data/${name}`;

function requireValue(value: unknown): asserts value {
  if (!value) throw new Error('Malformed recorded data: unsupported schema or invalid values.');
}

function object(value: unknown): Record<string, unknown> {
  requireValue(value && typeof value === 'object' && !Array.isArray(value));
  return value as Record<string, unknown>;
}

function list(value: unknown): unknown[] {
  requireValue(Array.isArray(value));
  return value as unknown[];
}

function finite(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function text(value: unknown): value is string {
  return typeof value === 'string' && value.length > 0;
}

function hashes(value: unknown) {
  const entries = Object.entries(object(value));
  requireValue(entries.length > 0 && entries.every(([path, hash]) =>
    text(path) && typeof hash === 'string' && /^[a-f0-9]{64}$/.test(hash)));
}

export function parseCircuits(value: unknown): CircuitData {
  const data = object(value);
  const metadata = object(data.metadata);
  requireValue(data.schema_version === 1 && metadata.backend === 'numpy_cpu' &&
    metadata.precision === 'complex128' && text(metadata.bit_order) &&
    text(metadata.python_version) && text(metadata.numpy_version));
  hashes(metadata.source_sha256);
  const circuits = list(data.circuits);
  requireValue(circuits.length === 3);
  circuits.forEach((value, index) => {
    const circuit = object(value);
    const n = [2, 3, 1][index];
    const count = [2, 3, 3][index];
    requireValue(circuit.id === ['bell', 'ghz', 'interference'][index] &&
      circuit.num_qubits === n && text(circuit.name));
    const recordings = list(circuit.recordings);
    requireValue(recordings.length === (index === 2 ? 33 : 1));
    recordings.forEach((value, angleIndex) => {
      const recording = object(value);
      requireValue(index === 2
        ? finite(recording.angle_radians) && Math.abs(recording.angle_radians - angleIndex * Math.PI / 16) < 1e-12
        : recording.angle_radians === null);
      const operations = list(recording.operations);
      requireValue(operations.length === count);
      operations.forEach((value, step) => {
        const operation = object(value);
        const phase = index === 2;
        requireValue(operation.name === (phase ? ['H', 'RZ', 'H'][step] : step === 0 ? 'H' : 'CX'));
        requireValue(operation.target === (phase ? 0 : step) &&
          operation.control === (phase || step === 0 ? null : step - 1) &&
          operation.angle === (phase && step === 1 ? recording.angle_radians : null));
      });
      const steps = list(recording.steps);
      requireValue(steps.length === count + 1);
      steps.forEach(value => {
        const step = object(value);
        for (const key of ['real', 'imag', 'probabilities']) {
          const values = list(step[key]);
          requireValue(values.length === 2 ** n && values.every(finite));
        }
        const probabilities = step.probabilities as number[];
        requireValue(probabilities.every(p => p >= 0 && p <= 1 + 1e-12) &&
          Math.abs(probabilities.reduce((a, b) => a + b, 0) - 1) < 1e-12);
      });
    });
  });
  return value as CircuitData;
}

export function parseBenchmark(value: unknown): BenchmarkData {
  const data = object(value);
  requireValue(data.schema_version === 1 && data.backend === 'numpy_cpu' &&
    data.dtype === 'complex128' && data.units === 'nanoseconds' &&
    data.metric === 'complete_cpu_single_gate_run_latency' &&
    data.percentile_method === 'linear' && finite(data.seed));
  const metadata = object(data.metadata);
  for (const key of ['timestamp_utc', 'cpu_model', 'platform', 'python_version',
    'numpy_version', 'quantaforge_version', 'git_revision']) requireValue(text(metadata[key]));
  requireValue(metadata.cpu_model === 'Apple M4 Pro' && typeof metadata.git_dirty === 'boolean');
  hashes(metadata.source_sha256);
  const cases = list(data.cases);
  requireValue(cases.length === 16);
  const keys = new Set<string>();
  cases.forEach(value => {
    const entry = object(value);
    requireValue(typeof entry.num_qubits === 'number' && [8, 12, 16, 18].includes(entry.num_qubits));
    requireValue(entry.target === 0 || entry.target === entry.num_qubits - 1);
    requireValue(entry.operation === 'H' || entry.operation === 'RX');
    requireValue(entry.operation === 'H' ? entry.angle_radians === null : finite(entry.angle_radians));
    requireValue(entry.warmups === 5 && entry.repetitions === 31);
    requireValue(finite(entry.median_ns) && entry.median_ns > 0 &&
      finite(entry.p95_ns) && entry.p95_ns >= entry.median_ns);
    const timings = list(entry.timings_ns);
    requireValue(timings.length === 31 && timings.every(n => finite(n) && n > 0));
    keys.add(`${entry.operation}-${entry.num_qubits}-${entry.target}`);
  });
  requireValue(keys.size === 16);
  return value as BenchmarkData;
}

export async function fetchRecorded<T>(filename: string, parse: (value: unknown) => T, signal: AbortSignal) {
  const response = await fetch(assetUrl(filename), { signal });
  if (!response.ok) throw new Error(`Could not load ${filename} (HTTP ${response.status}).`);
  return parse(await response.json());
}
