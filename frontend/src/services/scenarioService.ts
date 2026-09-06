import { SCENARIOS } from '../data/scenarios.ts';
import type { Scenario } from '../types.ts';

export interface ScenarioService {
  listScenarios(): readonly Scenario[];
  loadDemoScenario(id: string, signal?: AbortSignal): Promise<Scenario>;
}

export function listScenarios(): readonly Scenario[] {
  return SCENARIOS;
}

export async function loadDemoScenario(id: string, signal?: AbortSignal): Promise<Scenario> {
  const scenario = SCENARIOS.find((item) => item.id === id);
  if (!scenario) throw new Error('This scenario is unavailable. Please select a prepared example.');
  if (signal?.aborted) throw new DOMException('Scenario loading cancelled', 'AbortError');
  // This brief delay is presentation feedback only. No model inference occurs.
  await new Promise<void>((resolve, reject) => {
    const onAbort = () => {
      clearTimeout(timer);
      reject(new DOMException('Scenario loading cancelled', 'AbortError'));
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, 650);
    signal?.addEventListener('abort', onAbort, { once: true });
  });
  return structuredClone(scenario);
}

export const scenarioService: ScenarioService = { listScenarios, loadDemoScenario };
