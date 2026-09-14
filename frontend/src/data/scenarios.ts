import type { OceanLocation, Scenario } from '../types.ts';

export const DEPTHS = [0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500] as const;

export const LOCATIONS: OceanLocation[] = [
  { id: 'central', name: 'Central Bay', latitude: 14.5, longitude: 88, label: '01' },
  { id: 'northern', name: 'Northern Bay', latitude: 19, longitude: 89, label: '02' },
  { id: 'western', name: 'Western Bay', latitude: 13.5, longitude: 83.5, label: '03' },
  { id: 'andaman', name: 'Andaman Basin', latitude: 11.5, longitude: 94.5, label: '04' },
  { id: 'southern', name: 'Southern Bay', latitude: 7, longitude: 86, label: '05' },
];

// Authored demonstration fixtures, not downloaded observations or model predictions.
// Each tuple supplies an independently shaped profile for one location and season.
const FIXTURES: [number, number, number, number, number[], number[], string][] = [
  [0, 0, 0.12, 33.4, [29.6,29.5,29.1,27.8,24.9,21.5,17.4,14.8,12.1,10.6,9.2], [29.4,29.2,28.8,27.2,24.2,21.9,17.9,15.0,12.4,10.4,9.0], 'A warm surface layer gives way to a pronounced thermocline in this pre-monsoon illustration.'],
  [0, 1, 0.19, 32.8, [28.1,28.0,27.9,27.2,25.8,23.3,18.5,15.3,12.2,10.5,9.1], [28.3,28.2,28.1,26.8,25.1,22.8,18.9,15.7,12.0,10.7,9.4], 'This post-monsoon example has a deeper warm layer and a fresher surface than its pre-monsoon counterpart.'],
  [1, 0, 0.09, 32.1, [29.2,29.0,28.4,25.5,22.1,19.5,16.5,14.4,11.9,10.2,8.9], [29.0,28.7,27.9,24.9,22.6,19.9,16.2,14.8,12.2,10.0,9.2], 'The northern example places the strongest temperature change nearer the surface, beneath a warm cap.'],
  [1, 1, 0.24, 30.9, [27.5,27.4,27.0,25.9,23.7,21.1,17.8,15.1,12.0,10.4,9.0], [27.7,27.6,26.5,25.3,23.1,21.5,18.2,14.8,12.3,10.6,8.8], 'A fresher surface and a broader transition layer distinguish this synthetic northern post-monsoon scenario.'],
  [2, 0, -0.04, 34.0, [30.0,29.8,29.2,26.4,22.8,20.0,16.3,14.1,11.7,10.1,8.8], [29.8,29.5,28.9,25.8,22.2,20.4,16.7,14.4,11.5,10.3,9.0], 'This western example illustrates a shallower thermocline with relatively saline surface water.'],
  [2, 1, 0.08, 33.1, [28.0,27.9,27.8,27.4,25.3,22.0,17.9,14.9,12.3,10.5,9.2], [28.2,28.1,27.6,26.9,24.7,22.5,18.3,14.6,12.0,10.8,9.0], 'The post-monsoon western profile holds similar temperatures through its upper layer before cooling more quickly.'],
  [3, 0, 0.16, 33.0, [30.2,30.0,29.5,28.2,26.0,23.8,19.0,16.0,12.7,10.8,9.5], [30.0,29.8,29.2,27.6,25.5,23.3,19.4,16.4,12.4,11.0,9.2], 'This sheltered-basin illustration combines a warm surface with a gradual transition into deeper water.'],
  [3, 1, 0.21, 31.8, [28.8,28.7,28.5,28.0,26.8,24.6,20.1,16.5,12.8,10.9,9.6], [28.6,28.5,28.2,27.4,26.2,25.0,20.5,16.1,13.1,10.7,9.3], 'The synthetic Andaman post-monsoon profile shows a relatively deep warm layer and fresher surface water.'],
  [4, 0, 0.05, 34.5, [29.4,29.3,29.1,28.5,27.2,24.9,19.8,16.2,12.4,10.5,9.2], [29.2,29.0,28.8,28.0,26.6,24.4,20.2,16.6,12.2,10.8,9.0], 'A thick warm layer and more saline surface characterize this southern open-ocean example.'],
  [4, 1, 0.02, 34.1, [28.5,28.4,28.2,27.6,25.9,23.0,18.1,15.0,12.0,10.4,9.1], [28.7,28.6,27.9,27.1,25.3,23.4,18.5,14.7,12.3,10.2,8.9], 'This southern post-monsoon example cools through the upper 200 metres more quickly than its seasonal counterpart.'],
];

export const SCENARIOS: Scenario[] = FIXTURES.map(([locationIndex, seasonIndex, sla, sss, estimates, references, description]) => ({
  id: `${LOCATIONS[locationIndex].id}-${seasonIndex === 0 ? 'pre' : 'post'}`,
  location: LOCATIONS[locationIndex],
  date: seasonIndex === 0 ? '2024-04-15' : '2024-11-15',
  season: seasonIndex === 0 ? 'Pre-monsoon' : 'Post-monsoon',
  inputs: { sst: estimates[0], sla, sss, units: { sst: '°C', sla: 'm', sss: 'PSU' } },
  profile: DEPTHS.map((depth, i) => ({ depth, estimate: estimates[i], reference: references[i] })),
  description,
  provenance: 'synthetic',
}));

export const DEFAULT_SCENARIO = SCENARIOS[0];
