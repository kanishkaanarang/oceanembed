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
  windSpeed?: number | null;
  currentSpeed?: number | null;
  units: {
    sst: '°C';
    sla: 'm';
    sss: 'PSU';
    wind?: 'm/s';
    current?: 'm/s';
  };
}

export interface ProfilePoint {
  depth: number;
  estimate: number | null;     // Temperature in °C
  reference: number | null;    // Reference/ground truth in °C
  salinity?: number;           // Salinity in PSU
  density?: number;            // Seawater density in kg/m³
  soundSpeed?: number;         // Sound velocity in m/s (Mackenzie 1981)
}

export interface Scenario {
  id: string;
  location: OceanLocation;
  date: string;
  season: 'Pre-monsoon' | 'Post-monsoon' | 'Monsoon' | 'Winter';
  inputs: SurfaceInputs;
  profile: ProfilePoint[];
  description: string;
  provenance: 'synthetic' | 'oceanembed_cnn_v2' | 'analytical_ocean_physics';
  thermoclineDepth?: number;
  sofarAxisDepth?: number;
  hydrostaticStability?: 'stable' | 'inversion_warning';
  isLiveModel?: boolean;
}

export interface DepthMetric {
  depth_m: number;
  rmse_celsius: number;
  mae_celsius: number;
  status?: string;
}

export interface ModelBenchmarkMetrics {
  model_name?: string;
  overall_rmse: number;
  overall_mae: number;
  r2_score: number;
  baseline_rmse: number;
  depth_breakdown: DepthMetric[];
}
