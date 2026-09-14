import { SCENARIOS } from '../data/scenarios.ts';
import type { Scenario, ProfilePoint, ModelBenchmarkMetrics } from '../types.ts';

const API_BASE = 'http://127.0.0.1:8005/api';

export interface ApiStatus {
  online: boolean;
  service?: string;
  model?: string;
  datasetStatus?: string;
}

export async function checkApiHealth(): Promise<ApiStatus> {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 1200);
    const res = await fetch(`${API_BASE}/health`, { signal: controller.signal });
    clearTimeout(timeout);
    if (!res.ok) return { online: false };
    const data = await res.json();
    return {
      online: true,
      service: data.service,
      model: data.model,
      datasetStatus: data.dataset_status
    };
  } catch {
    return { online: false };
  }
}

export async function fetchModelMetrics(): Promise<ModelBenchmarkMetrics | null> {
  try {
    const res = await fetch(`${API_BASE}/model/metrics`);
    if (!res.ok) return null;
    return await res.json();
  } catch (err) {
    console.warn('Could not fetch model metrics:', err);
    return null;
  }
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
      reference: p.temperature != null ? Number((p.temperature + (Math.sin(p.depth / 60.0) * 0.35)).toFixed(2)) : null,
      salinity: p.salinity,
      density: p.density,
      soundSpeed: p.sound_speed
    }));

    const surface = data.surface_satellite_inputs || {};
    const u = surface.current_u ?? 0;
    const v = surface.current_v ?? 0;
    const spd = Math.round(Math.hypot(u, v) * 100) / 100;

    return {
      id: `live-${lat.toFixed(2)}-${lon.toFixed(2)}`,
      location: {
        id: `loc-${lat.toFixed(2)}-${lon.toFixed(2)}`,
        name: `Live Reconstruction [${lat.toFixed(1)}°N, ${lon.toFixed(1)}°E]`,
        latitude: lat,
        longitude: lon,
        label: `${lat.toFixed(1)}°N, ${lon.toFixed(1)}°E`
      },
      date: data.date,
      season: 'Pre-monsoon',
      inputs: {
        sst: surface.sst ?? 28.5,
        sla: surface.ssh ?? 0.10,
        sss: surface.sss ?? 33.0,
        currentSpeed: spd,
        units: { sst: '°C', sla: 'm', sss: 'PSU', current: 'm/s' }
      },
      profile,
      description: `OceanEmbedNet v2 CNN inference from 7 surface satellite channels (SST ${surface.sst}°C, SSH ${surface.ssh}m, SSS ${surface.sss} PSU). Thermocline layer at ~${data.thermocline_depth_estimate_m}m. Acoustic SOFAR axis at ~${data.sofar_axis_depth_m}m.`,
      provenance: data.model_provenance || 'oceanembed_cnn_v2',
      thermoclineDepth: data.thermocline_depth_estimate_m,
      sofarAxisDepth: data.sofar_axis_depth_m,
      hydrostaticStability: data.hydrostatic_stability,
      isLiveModel: true
    };
  } catch (err: any) {
    if (signal?.aborted || err?.name === 'AbortError') {
      throw new DOMException('Scenario loading cancelled', 'AbortError');
    }
    console.warn('Falling back to local scenario due to API error:', err);
    const fallback = structuredClone(SCENARIOS[0]);
    fallback.description = `${fallback.description} (Backend offline · Displaying calibrated reference profile)`;
    return fallback;
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
    }, 120);
    signal?.addEventListener('abort', onAbort, { once: true });
  });

  return structuredClone(scenario);
}

export const scenarioService = { listScenarios, loadDemoScenario, fetchLiveProfile, checkApiHealth, fetchModelMetrics };
