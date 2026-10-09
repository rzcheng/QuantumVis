export interface Operation {
  name: 'H' | 'RZ' | 'CX';
  target: number;
  control: number | null;
  angle: number | null;
}

export interface Snapshot {
  real: number[];
  imag: number[];
  probabilities: number[];
}

export interface Recording {
  angle_radians: number | null;
  operations: Operation[];
  steps: Snapshot[];
}

export interface RecordedCircuit {
  id: 'bell' | 'ghz' | 'interference';
  name: string;
  num_qubits: number;
  recordings: Recording[];
}

export interface CircuitData {
  schema_version: 1;
  metadata: {
    backend: 'numpy_cpu';
    precision: 'complex128';
    bit_order: string;
    python_version: string;
    numpy_version: string;
    source_sha256: Record<string, string>;
  };
  circuits: RecordedCircuit[];
}

export interface BenchmarkCase {
  num_qubits: number;
  target: number;
  operation: 'H' | 'RX';
  angle_radians: number | null;
  warmups: number;
  repetitions: number;
  timings_ns: number[];
  median_ns: number;
  p95_ns: number;
}

export interface BenchmarkData {
  schema_version: 1;
  backend: 'numpy_cpu';
  dtype: 'complex128';
  units: 'nanoseconds';
  metric: string;
  percentile_method: string;
  seed: number;
  metadata: {
    timestamp_utc: string;
    cpu_model: string;
    platform: string;
    python_version: string;
    numpy_version: string;
    quantaforge_version: string;
    git_revision: string;
    git_dirty: boolean;
    source_sha256: Record<string, string>;
  };
  cases: BenchmarkCase[];
}
