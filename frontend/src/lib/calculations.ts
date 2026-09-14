import type { ProfilePoint, Scenario } from '../types.ts';

export function formatValue(value: number | null | undefined, digits = 1): string {
  return value != null && Number.isFinite(value) ? value.toFixed(digits) : 'Unavailable';
}

export function calculateAgreement(profile: ProfilePoint[]) {
  const pairs = profile.filter((point) => point.estimate != null && point.reference != null && Number.isFinite(point.estimate) && Number.isFinite(point.reference));
  if (!pairs.length) return null;
  let absolute = 0;
  let squared = 0;
  for (const point of pairs) {
    const error = point.estimate! - point.reference!;
    absolute += Math.abs(error);
    squared += error ** 2;
  }
  return { mae: absolute / pairs.length, rmse: Math.sqrt(squared / pairs.length), count: pairs.length };
}

export function profileInterpretation(scenario: Scenario): string {
  const surface = scenario.profile.find((point) => point.depth === 0)?.estimate;
  const bottom = scenario.profile.find((point) => point.depth === 500)?.estimate;
  if (surface == null || bottom == null) return 'Some temperatures are unavailable in this scenario.';
  let strongest: { start: number; end: number; gradient: number } | null = null;
  for (let i = 1; i < scenario.profile.length; i++) {
    const previous = scenario.profile[i - 1];
    const current = scenario.profile[i];
    if (previous.estimate == null || current.estimate == null || current.depth <= previous.depth) continue;
    const gradient = (previous.estimate - current.estimate) / (current.depth - previous.depth);
    if (!strongest || gradient > strongest.gradient) strongest = { start: previous.depth, end: current.depth, gradient };
  }
  return `Temperature falls from ${surface.toFixed(1)}°C at the surface to ${bottom.toFixed(1)}°C at 500 m.${strongest ? ` The steepest cooling is between ${strongest.start} and ${strongest.end} m.` : ''}`;
}

export function buildScenarioCsv(scenario: Scenario): string {
  const fields = ['scenario_id', 'location', 'latitude_deg', 'longitude_deg', 'date', 'season', 'depth_m', 'illustrative_estimate_degC', 'synthetic_reference_degC', 'sst_degC', 'sla_m', 'sss_PSU', 'provenance'];
  const escape = (value: unknown) => `"${String(value ?? '').replaceAll('"', '""')}"`;
  const rows = scenario.profile.map((point) => [scenario.id, scenario.location.name, scenario.location.latitude, scenario.location.longitude, scenario.date, scenario.season, point.depth, point.estimate, point.reference, scenario.inputs.sst, scenario.inputs.sla, scenario.inputs.sss, 'synthetic; illustrative output; no trained model']);
  return [fields, ...rows].map((row) => row.map(escape).join(',')).join('\r\n') + '\r\n';
}

export function downloadScenario(scenario: Scenario) {
  const url = URL.createObjectURL(new Blob([buildScenarioCsv(scenario)], { type: 'text/csv;charset=utf-8;' }));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `oceanembed-${scenario.id}-${scenario.date}-synthetic.csv`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function displayDate(date: string) {
  return new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' }).format(new Date(`${date}T12:00:00Z`));
}
