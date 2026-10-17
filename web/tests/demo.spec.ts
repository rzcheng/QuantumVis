import { expect, test, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import circuitData from '../public/data/circuits.json' with { type: 'json' };
import benchmarkData from '../public/data/cpu-baseline.json' with { type: 'json' };

async function open(page: Page) {
  await page.goto('./');
  await expect(page.getByRole('table', { name: 'Recorded state vector' })).toBeVisible();
  await expect(page.getByRole('table', { name: 'H recorded latency in ms' })).toBeVisible();
}

test('default Bell final state includes every basis state and undefined zero phases', async ({ page }) => {
  await open(page);
  await expect(page.getByLabel('Preset')).toHaveValue('0');
  await expect(page.getByRole('status').filter({ hasText: 'step' })).toHaveText(/step 2 of 2/);
  await expect(page.getByRole('button', { name: 'Next', exact: true })).toBeDisabled();
  await expect(page.locator('.basis')).toHaveText(['|00⟩', '|01⟩', '|10⟩', '|11⟩']);
  await expect(page.locator('.probability > span')).toHaveText(['50.0%', '0.0%', '0.0%', '50.0%']);
  await expect(page.getByLabel('Phase undefined at zero amplitude')).toHaveCount(2);
});

for (const [preset, last] of [[0, 2], [1, 3], [2, 3]]) {
  test(`preset ${preset} resets to its final state and respects every step boundary`, async ({ page }) => {
    await open(page);
    await page.getByRole('button', { name: 'Initial', exact: true }).click();
    if (preset === 0) await page.getByLabel('Preset').selectOption('1');
    await page.getByLabel('Preset').selectOption(String(preset));
    await expect(page.getByRole('status').filter({ hasText: 'step' })).toHaveText(new RegExp(`step ${last} of ${last}`));
    await page.getByRole('button', { name: 'Initial', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Initial', exact: true })).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Previous', exact: true })).toBeDisabled();
    await expect(page.locator('.probability > span').first()).toHaveText('100.0%');
    for (let step = 1; step <= last; step++) {
      await page.getByRole('button', { name: 'Next', exact: true }).click();
      await expect(page.getByRole('status').filter({ hasText: 'step' })).toHaveText(new RegExp(`step ${step} of ${last}`));
      await expect(page.getByRole('button', { name: new RegExp(`^Step ${step}:`) })).toHaveAttribute('aria-pressed', 'true');
    }
    await expect(page.getByRole('button', { name: 'Next', exact: true })).toBeDisabled();
    for (let step = last - 1; step >= 0; step--) {
      await page.getByRole('button', { name: 'Previous', exact: true }).click();
      await expect(page.getByRole('status').filter({ hasText: 'step' })).toHaveText(new RegExp(`step ${step} of ${last}`));
    }
  });
}

test('gate selection is keyboard accessible and selects the state after that gate', async ({ page }) => {
  await open(page);
  const h = page.getByRole('button', { name: 'Step 1: H on q0' });
  await h.focus();
  await expect(h).toBeFocused();
  await h.press('Enter');
  await expect(page.locator('.probability > span')).toHaveText(['50.0%', '50.0%', '0.0%', '0.0%']);
  const cx = page.getByRole('button', { name: 'Step 2: CX q0 → q1' });
  await cx.press('Space');
  await expect(cx).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('.gate.completed')).toHaveCount(1);
  await h.click();
  await expect(page.locator('.gate.upcoming')).toHaveCount(1);
});

test('all 33 angle settings select exported snapshots with inclusive endpoints', async ({ page }) => {
  await open(page);
  await page.getByLabel('Preset').selectOption('2');
  const slider = page.getByLabel('recorded angle', { exact: true });
  await expect(slider).toHaveValue('8');
  await expect(page.locator('output')).toContainText('π/2');
  await slider.press('Home');
  for (let index = 0; index < 33; index++) {
    await expect(slider).toHaveValue(String(index));
    const recording = circuitData.circuits[2].recordings[index];
    await expect(page.locator('.probability > span')).toHaveText(
      recording.steps[3].probabilities.map(p => `${(p * 100).toFixed(1)}%`));
    if (index < 32) await slider.press('ArrowRight');
  }
  await slider.press('ArrowRight');
  await expect(slider).toHaveValue('32');
  await expect(page.locator('output')).toContainText('2π');
  await slider.press('Home');
  await slider.press('ArrowLeft');
  await expect(slider).toHaveValue('0');
});

test('RZ preserves probabilities, changes phase, and pi interference displays minus i', async ({ page }) => {
  await open(page);
  await page.getByLabel('Preset').selectOption('2');
  await page.getByLabel('recorded angle', { exact: true }).fill('16');
  await page.getByRole('button', { name: 'Step 1: H on q0' }).click();
  await expect(page.locator('.probability > span')).toHaveText(['50.0%', '50.0%']);
  await page.getByRole('button', { name: 'Step 2: RZ on q0' }).click();
  await expect(page.locator('.probability > span')).toHaveText(['50.0%', '50.0%']);
  await expect(page.getByRole('img', { name: 'Phase -90.0 degrees', exact: true })).toBeVisible();
  await expect(page.getByRole('img', { name: 'Phase 90.0 degrees', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Next', exact: true }).click();
  await expect(page.locator('.amplitude').nth(1)).toContainText('0.0000 − 1.0000i');
  await expect(page.locator('.probability > span')).toHaveText(['0.0%', '100.0%']);
  await page.getByLabel('Preset').selectOption('0');
  await expect(page.getByLabel('recorded angle', { exact: true })).toHaveCount(0);
  await page.getByLabel('Preset').selectOption('2');
  await expect(page.getByLabel('recorded angle', { exact: true })).toHaveValue('16');
});

for (const target of [0, 1, 2]) {
  test(`target q${target} pairs uniquely cover all eight indices and differ by the correct bit`, async ({ page }) => {
    await open(page);
    await page.getByLabel('Target', { exact: true }).selectOption(String(target));
    const seen = new Set<number>();
    for (let pair = 0; pair < 4; pair++) {
      const radio = page.getByRole('radio', { name: new RegExp(`^Pair ${pair}`) });
      await radio.check();
      const selected = await page.locator('.storage-row').first().locator('.selected-cell').allTextContents();
      const indices = selected.map(text => Number(text.split('i')[0]));
      expect(indices).toHaveLength(2);
      expect(indices[0] ^ indices[1]).toBe(1 << target);
      expect(indices[0] & (1 << target)).toBe(0);
      for (const index of indices) {
        expect(seen.has(index)).toBe(false);
        seen.add(index);
        await expect(page.getByLabel(`real index ${index}, selected`, { exact: true })).toBeVisible();
        await expect(page.getByLabel(`imaginary index ${index}, selected`, { exact: true })).toBeVisible();
        await expect(page.locator('.binary-labels')).toContainText(index.toString(2).padStart(3, '0'));
      }
      await expect(page.locator('.selected-cell')).toHaveCount(4);
    }
    expect([...seen].sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
  });
}

test('pair controls work by keyboard and target changes reset pair zero', async ({ page }) => {
  await open(page);
  await expect(page.getByLabel('Target', { exact: true })).toHaveValue('0');
  const radios = page.getByRole('radio');
  await radios.nth(0).focus();
  await radios.nth(0).press('ArrowRight');
  await expect(radios.nth(1)).toBeChecked();
  await page.getByLabel('Target', { exact: true }).selectOption('2');
  await expect(radios.nth(0)).toBeChecked();
  await expect(page.locator('.binary-labels')).toHaveText('|000⟩ ↔ |100⟩');
});

for (const operation of ['H', 'RX']) {
  test(`${operation} selector and numerical table convert the original nanoseconds to ms`, async ({ page }) => {
    await open(page);
    await expect(page.getByLabel('Gate', { exact: true })).toHaveValue('H');
    await page.getByLabel('Gate', { exact: true }).selectOption(operation);
    const table = page.getByRole('table', { name: new RegExp(`^${operation} recorded latency`) });
    const cases = benchmarkData.cases.filter(entry => entry.operation === operation);
    await expect(table.locator('tbody tr')).toHaveCount(8);
    for (const [index, entry] of cases.entries()) {
      await expect(table.locator('tbody tr').nth(index)).toHaveText(
        `${entry.num_qubits}${entry.target === 0 ? '0' : `${entry.target} (n−1)`}${(entry.median_ns / 1e6).toFixed(6)}${(entry.p95_ns / 1e6).toFixed(6)}`);
    }
    await expect(page.locator('.median-line')).toHaveCount(2);
    await expect(page.locator('.percentile-marker')).toHaveCount(8);
    await expect(page.locator('.benchmark-plot')).toHaveAccessibleName(/logarithmic y-axis/);
    await expect(page.locator('.benchmark-plot .tick')).toHaveText(['0.01', '0.1', '1', '10', '8', '12', '16', '18']);
  });
}

test('nested assets load cleanly and raw download is byte-identical', async ({ page, request }, testInfo) => {
  const errors: string[] = [];
  const assets: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  page.on('response', response => {
    if (response.url().startsWith('http://127.0.0.1')) {
      assets.push(response.url());
      if (response.status() >= 400) errors.push(`${response.status()} ${response.url()}`);
    }
  });
  await open(page);
  await page.getByText('Recording details & raw data', { exact: true }).click();
  const link = page.getByRole('link', { name: 'Download raw benchmark JSON' });
  await expect(link).toHaveAttribute('href', '/QuantumVis/data/cpu-baseline.json');
  const downloadPromise = page.waitForEvent('download');
  await link.click();
  const download = await downloadPromise;
  const path = testInfo.outputPath('download.json');
  await download.saveAs(path);
  const original = await readFile('../benchmarks/results/cpu-m4-pro-2026-09-14.json');
  expect(await readFile(path)).toEqual(original);
  const response = await request.get('data/circuits.json');
  expect(response.status()).toBe(200);
  expect(await response.json()).toEqual(circuitData);
  expect((await request.get('missing-route')).status()).toBe(404);
  expect((await request.get('data/missing.json')).status()).toBe(404);
  expect(assets.some(url => url.includes('/QuantumVis/assets/') && url.endsWith('.js'))).toBe(true);
  expect(assets.filter(url => /\.(json|js|css)$/.test(url)).every(url => url.includes('/QuantumVis/'))).toBe(true);
  expect(errors).toEqual([]);
  await page.getByText('Recorded source SHA-256 hashes', { exact: true }).click();
  await expect(page.locator('.source-hashes')).toContainText(benchmarkData.metadata.source_sha256['benchmarks/benchmark_gates.py']);
});

test('loading is explicit while data is in flight', async ({ page }) => {
  let release!: () => void;
  const waiting = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/data/circuits.json', async route => { await waiting; await route.continue(); });
  await page.goto('./');
  await expect(page.getByRole('region', { name: 'Circuit explorer' }).getByRole('status')).toHaveText('Loading recorded data…');
  await expect(page.getByLabel('Preset')).toHaveCount(0);
  release();
  await expect(page.getByLabel('Preset')).toBeVisible();
});

for (const filename of ['circuits.json', 'cpu-baseline.json']) {
  for (const failure of ['missing', 'malformed', 'invalid-json']) {
    test(`${filename}: ${failure} shows an error without fake values and retry recovers`, async ({ page }, testInfo) => {
      await page.route(`**/data/${filename}`, route => route.fulfill({
        status: failure === 'missing' ? 404 : 200,
        contentType: 'application/json',
        body: failure === 'malformed' ? '{"schema_version":999}' : 'invalid json',
      }));
      await page.goto('./');
      await expect(page.getByRole('alert')).toContainText('Recorded data unavailable.');
      await expect(page.getByLabel(filename === 'circuits.json' ? 'Preset' : 'Gate', { exact: true })).toHaveCount(0);
      if (filename === 'circuits.json' && failure === 'malformed') await page.screenshot({ path: testInfo.outputPath('error.png') });
      await page.unroute(`**/data/${filename}`);
      await page.getByRole('button', { name: 'Retry loading' }).click();
      await expect(page.getByRole('alert')).toHaveCount(0);
      await expect(page.getByLabel(filename === 'circuits.json' ? 'Preset' : 'Gate', { exact: true })).toBeVisible();
    });
  }
}

test('invalid snapshot shape and duplicate benchmark cases fail closed', async ({ page }) => {
  const circuit = structuredClone(circuitData);
  circuit.circuits[0].recordings[0].steps[0].real.pop();
  const benchmark = structuredClone(benchmarkData);
  benchmark.cases[1] = benchmark.cases[0];
  await page.route('**/data/circuits.json', route => route.fulfill({ json: circuit }));
  await page.route('**/data/cpu-baseline.json', route => route.fulfill({ json: benchmark }));
  await page.goto('./');
  await expect(page.getByRole('alert')).toHaveCount(2);
  await expect(page.getByRole('table')).toHaveCount(0);
});

for (const width of [1280, 390, 360]) {
  test(`layout at ${width}px keeps controls and data within the page`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: 900 });
    await open(page);
    await expect(page.getByRole('table', { name: 'Recorded state vector' }).getByRole('columnheader')).toHaveCount(4);
    const plot = await page.locator('.benchmark-plot').boundingBox();
    for (const tick of await page.locator('.benchmark-plot .tick').all()) {
      const box = (await tick.boundingBox())!;
      expect(box.x).toBeGreaterThanOrEqual(plot!.x);
      expect(box.x + box.width).toBeLessThanOrEqual(plot!.x + plot!.width);
      expect(box.height).toBeGreaterThanOrEqual(12);
    }
    await page.screenshot({ path: testInfo.outputPath(`bell-${width}.png`), fullPage: true });
    await page.getByLabel('Preset').selectOption('1');
    await expect(page.locator('.state-row')).toHaveCount(8);
    await page.getByLabel('Target', { exact: true }).selectOption('2');
    await page.getByRole('radio', { name: /^Pair 3/ }).check();
    await page.getByLabel('Gate', { exact: true }).selectOption('RX');
    await page.getByText('Recording details & raw data', { exact: true }).click();
    await page.getByText('Recorded source SHA-256 hashes', { exact: true }).click();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    await page.screenshot({ path: testInfo.outputPath(`ghz-details-${width}.png`), fullPage: true });
    await page.getByLabel('Preset').selectOption('2');
    await page.getByLabel('recorded angle', { exact: true }).fill('16');
    await page.getByText('Reading phase and the recorded data', { exact: true }).click();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    await page.screenshot({ path: testInfo.outputPath(`interference-${width}.png`), fullPage: true });
  });
}

test('section links, keyboard focus, reduced motion, and visible text contrast', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await open(page);
  await page.keyboard.press('Tab');
  await expect(page.getByRole('link', { name: 'Skip to content' })).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.locator('main')).toBeFocused();
  for (const [name, id] of [['Circuit', 'circuit'], ['Kernel', 'kernel'], ['Benchmarks', 'benchmarks']]) {
    await page.getByRole('link', { name, exact: true }).click();
    await expect(page).toHaveURL(new RegExp(`#${id}$`));
  }
  await page.getByRole('link', { name: 'QuantumVis', exact: true }).click();
  await expect(page).toHaveURL(/#main$/);
  await page.getByLabel('Preset').focus();
  const style = await page.getByLabel('Preset').evaluate(element => {
    const style = getComputedStyle(element);
    return { outline: style.outlineWidth, transition: style.transitionDuration };
  });
  expect(style.outline).toBe('2px');
  expect(style.transition).toBe('0s');
  const luminance = (hex: string) => {
    const rgb = hex.match(/\w\w/g)!.map(n => parseInt(n, 16) / 255)
      .map(n => n <= 0.04045 ? n / 12.92 : ((n + 0.055) / 1.055) ** 2.4);
    return rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722;
  };
  for (const background of ['141719', '1b2023', '23332f']) {
    for (const foreground of ['e4e8e9', 'a4adb3', '83c5ba', '91afd2']) {
      expect((luminance(foreground) + 0.05) / (luminance(background) + 0.05)).toBeGreaterThan(4.5);
    }
  }
});
