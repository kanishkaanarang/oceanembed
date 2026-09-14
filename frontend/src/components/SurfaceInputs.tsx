import { Droplets, Thermometer, Waves } from 'lucide-react';
import type { SurfaceInputs as Inputs } from '../types';
import { formatValue } from '../lib/calculations';

export function SurfaceInputs({ inputs, compact = false }: { inputs: Inputs; compact?: boolean }) {
  const items = [
    { key: 'sst', name: 'Surface temperature', short: 'SST', value: inputs.sst, unit: inputs.units.sst, icon: Thermometer, digits: 1 },
    { key: 'sla', name: 'Sea-level anomaly', short: 'SLA', value: inputs.sla, unit: inputs.units.sla, icon: Waves, digits: 2 },
    { key: 'sss', name: 'Surface salinity', short: 'SSS', value: inputs.sss, unit: inputs.units.sss, icon: Droplets, digits: 1 },
  ];
  return <div className={`surface-inputs ${compact ? 'compact' : ''}`}>
    {items.map(({ key, name, short, value, unit, icon: Icon, digits }) => <div className={`input-card input-${key}`} key={key}>
      <div className="input-card-top"><Icon size={17} aria-hidden="true" /><span>{short}</span></div>
      <div className="input-value">{formatValue(value, digits)} <span>{unit}</span></div>
      <div className="input-name">{name}</div>
    </div>)}
  </div>;
}
