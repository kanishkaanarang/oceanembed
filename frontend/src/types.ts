export interface OceanLocation {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  label: string;
}

export interface SurfaceInputs {
  sst: number | null;
  sla: number | null;
  sss: number | null;
  units: { sst: '°C'; sla: 'm'; sss: 'PSU' };
}

export interface ProfilePoint {
  depth: number;
  estimate: number | null;
  reference: number | null;
}

export interface Scenario {
  id: string;
  location: OceanLocation;
  date: string;
  season: 'Pre-monsoon' | 'Post-monsoon';
  inputs: SurfaceInputs;
  profile: ProfilePoint[];
  description: string;
  provenance: 'synthetic';
}
