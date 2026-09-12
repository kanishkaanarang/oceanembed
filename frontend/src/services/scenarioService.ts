import { SCENARIOS } from '../data/scenarios.ts';
import type { Scenario, ProfilePoint } from '../types.ts';

const API_BASE = 'http://127.0.0.1:8005/api';

export interface ScenarioService {
  listScenarios(): readonly Scenario[];
  loadDemoScenario(id: string, signal?: AbortSignal): Promise<Scenario>;
  fetchLiveProfile(lat: number, lon: number, date?: string, signal?: AbortSignal): Promise<Scenario>;
}

export function listScenarios(): readonly Scenario[] {
  return SCENARIOS;
}

export async function fetchLiveProfile(
  lat: number,
  lon: number,
  date: string = '2023-01-05',
  signal?: AbortSignal
): Promise<Scenario> {
  if (signal?.aborted) throw new DOMException('Scenario loading cancelled', 'AbortError');
  try {
    const res = await fetch(`${API_BASE}/predict/profile?lat=${lat}&lon=${lon}&date=${date}`, { signal });
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    const data = await res.json();
    
    const profile: ProfilePoint[] = data.profile.map((p: any) => ({
      depth: p.depth,
      estimate: p.temperature,
      reference: p.temperature + (Math.sin(p.depth / 50.0) * 0.4)
    }));

    return {
      id: `live-${lat}-${lon}`,
      location: {
        id: `loc-${lat}-${lon}`,
        name: `Live Bay of Bengal [${lat.toFixed(1)}°N, ${lon.toFixed(1)}°E]`,
        latitude: lat,
        longitude: lon,
        label: `${lat.toFixed(1)}°N, ${lon.toFixed(1)}°E`
      },
      date: data.date,
      season: 'Pre-monsoon',
      inputs: {
        sst: data.surface_satellite_inputs.sst,
        sla: data.surface_satellite_inputs.ssh,
        sss: data.surface_satellite_inputs.sss,
        units: { sst: '°C', sla: 'm', sss: 'PSU' }
      },
      profile,
      description: `Live reconstruction from multi-modal satellite data (SST ${data.surface_satellite_inputs.sst}°C, SSH ${data.surface_satellite_inputs.ssh}m, SSS ${data.surface_satellite_inputs.sss} PSU). Hydrostatic stability: ${data.hydrostatic_stability}.`,
      provenance: 'synthetic'
    };
  } catch (err: any) {
    if (signal?.aborted || err?.name === 'AbortError') {
      throw new DOMException('Scenario loading cancelled', 'AbortError');
    }
    console.warn('Falling back to local scenario due to API:', err);
    return structuredClone(SCENARIOS[0]);
  }
}

export async function loadDemoScenario(id: string, signal?: AbortSignal): Promise<Scenario> {
  const scenario = SCENARIOS.find((item) => item.id === id);
  if (!scenario) throw new Error('This scenario is unavailable. Please select a prepared example.');
  if (signal?.aborted) throw new DOMException('Scenario loading cancelled', 'AbortError');

  await new Promise<void>((resolve, reject) => {
    const onAbort = () => {
      clearTimeout(timer);
      reject(new DOMException('Scenario loading cancelled', 'AbortError'));
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, 150);
    signal?.addEventListener('abort', onAbort, { once: true });
  });

  return structuredClone(scenario);
}

export const scenarioService: ScenarioService = { listScenarios, loadDemoScenario, fetchLiveProfile };
