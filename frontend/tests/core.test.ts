import assert from 'node:assert/strict';
import test from 'node:test';
import { DEPTHS, SCENARIOS } from '../src/data/scenarios.ts';
import { buildScenarioCsv, calculateAgreement, formatValue, profileInterpretation } from '../src/lib/calculations.ts';
import { loadDemoScenario } from '../src/services/scenarioService.ts';

test('all ten scenarios are complete, uniquely identified and explicitly synthetic', () => {
  assert.equal(SCENARIOS.length, 10);
  assert.equal(new Set(SCENARIOS.map((s) => s.id)).size, 10);
  for (const scenario of SCENARIOS) {
    assert.equal(scenario.provenance, 'synthetic');
    assert.deepEqual(scenario.profile.map((p) => p.depth), [...DEPTHS]);
    assert.equal(scenario.inputs.sst, scenario.profile[0].estimate);
    assert.ok(scenario.profile.every((p) => p.estimate != null && p.reference != null && Number.isFinite(p.estimate) && Number.isFinite(p.reference)));
    assert.equal(SCENARIOS.filter((s) => s.location.id === scenario.location.id).length, 2);
  }
});

test('agreement uses valid paired depths and handles missing or non-finite values', () => {
  const metrics = calculateAgreement([
    { depth: 0, estimate: 5, reference: 2 },
    { depth: 10, estimate: 4, reference: 0 },
    { depth: 25, estimate: null, reference: 12 },
    { depth: 50, estimate: 3, reference: Number.NaN },
  ]);
  assert.ok(metrics);
  assert.equal(metrics.count, 2);
  assert.equal(metrics.mae, 3.5);
  assert.equal(metrics.rmse, Math.sqrt(12.5));
  assert.equal(calculateAgreement([]), null);
  assert.equal(formatValue(null), 'Unavailable');
});

test('CSV preserves coordinates, depth units, temperatures and provenance for every row', () => {
  const scenario = SCENARIOS[0];
  const csv = buildScenarioCsv(scenario);
  const lines = csv.trim().split('\r\n');
  assert.equal(lines.length, 12);
  assert.match(lines[0], /depth_m/);
  assert.match(lines[0], /synthetic_reference_degC/);
  assert.match(lines[0], /sst_degC/);
  assert.ok(lines.slice(1).every((line) => line.includes('synthetic; illustrative output; no trained model')));
  assert.ok(lines[1].includes('"14.5","88","2024-04-15"'));
  assert.ok(lines[11].includes('"500","9.2","9"'));
});

test('CSV escapes quotes and commas, and leaves missing numbers blank', () => {
  const scenario = structuredClone(SCENARIOS[0]);
  scenario.location.name = 'Bay, "example"';
  scenario.profile[0].reference = null;
  const csv = buildScenarioCsv(scenario);
  assert.ok(csv.includes('"Bay, ""example"""'));
  assert.ok(csv.includes('"0","29.6",""'));
});

test('interpretation derives cooling gradient from actual depth intervals', () => {
  assert.match(profileInterpretation(SCENARIOS[0]), /29.6°C/);
  assert.match(profileInterpretation(SCENARIOS[0]), /9.2°C/);
  assert.match(profileInterpretation(SCENARIOS[0]), /75 and 100 m/);
});

test('unknown scenarios fail clearly', async () => {
  await assert.rejects(loadDemoScenario('missing-scenario'), /unavailable/);
});

test('in-flight scenario loads can be cancelled, including before loading starts', async () => {
  const controller = new AbortController();
  const loading = loadDemoScenario(SCENARIOS[0].id, controller.signal);
  controller.abort();
  await assert.rejects(loading, { name: 'AbortError' });
  await assert.rejects(loadDemoScenario(SCENARIOS[0].id, controller.signal), { name: 'AbortError' });
});

test('loaded scenarios cannot mutate the source fixtures', async () => {
  const loaded = await loadDemoScenario(SCENARIOS[0].id);
  loaded.profile[0].estimate = -100;
  assert.equal(SCENARIOS[0].profile[0].estimate, 29.6);
});
