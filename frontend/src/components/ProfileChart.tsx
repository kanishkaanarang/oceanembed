import { CartesianGrid, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts';
import type { Scenario } from '../types';
import { formatValue } from '../lib/calculations';

interface Props { scenario: Scenario; secondScenario?: Scenario; showReference?: boolean; selectedDepth?: number; }

export function ProfileChart({ scenario, secondScenario, showReference = true, selectedDepth = 100 }: Props) {
  const primary = scenario.profile.map((p) => ({ depth: p.depth, temperature: p.estimate }));
  const secondary = secondScenario ? secondScenario.profile.map((p) => ({ depth: p.depth, temperature: p.estimate })) : scenario.profile.map((p) => ({ depth: p.depth, temperature: p.reference }));
  const primaryName = secondScenario ? `${scenario.location.name} · ${scenario.season}` : 'Illustrative estimate';
  const secondaryName = secondScenario ? `${secondScenario.location.name} · ${secondScenario.season}` : 'Synthetic reference';
  const active = scenario.profile.find((p) => p.depth === selectedDepth);
  return <div className="profile-visual">
    <div className="chart-axis-title">TEMPERATURE (°C)</div>
    <div className="profile-chart" role="img" aria-label={`Temperature versus depth for ${scenario.location.name}. Depth increases downwards from 0 to 500 metres. ${primaryName}${showReference || secondScenario ? ` compared with ${secondaryName}` : ''}. A full data table follows.`}>
      <ResponsiveContainer width="100%" height="100%" minWidth={0}>
        <ScatterChart margin={{ top: 16, right: 17, bottom: 7, left: 3 }}>
          <CartesianGrid stroke="#263741" strokeDasharray="3 5" horizontal vertical />
          <XAxis type="number" dataKey="temperature" name="Temperature" unit="°C" domain={[5,35]} ticks={[5,10,15,20,25,30,35]} orientation="top" tick={{ fill: '#a2b5c0', fontSize: 11 }} axisLine={false} tickLine={false} tickMargin={9} />
          <YAxis type="number" dataKey="depth" name="Depth" unit=" m" domain={[0,500]} ticks={[0,100,200,300,400,500]} reversed tick={{ fill: '#a2b5c0', fontSize: 11 }} axisLine={false} tickLine={false} width={53} />
          <ReferenceLine y={selectedDepth} stroke="#6bbcaf" strokeDasharray="4 4" strokeOpacity={.7} />
          <Tooltip cursor={{ stroke: '#638c9c', strokeDasharray: '3 3' }} content={({ active: hovered, payload }) => {
            if (!hovered || !payload?.length) return null;
            const depth = payload[0]?.payload?.depth as number | undefined;
            const point = scenario.profile.find((p) => p.depth === depth);
            const other = secondScenario?.profile.find((p) => p.depth === depth);
            if (!point) return null;
            const secondValue = secondScenario ? other?.estimate : point.reference;
            const delta = point.estimate != null && secondValue != null ? Math.abs(point.estimate-secondValue) : null;
            return <div className="chart-tooltip"><strong>{depth} m below surface</strong><span>{primaryName}: {formatValue(point.estimate)}°C</span>{showReference || secondScenario ? <><span>{secondaryName}: {formatValue(secondValue)}°C</span><span>Absolute difference: {formatValue(delta,2)}°C</span></> : null}</div>;
          }} />
          {showReference || secondScenario ? <Scatter name={secondaryName} data={secondary} fill="#ec9b87" line={{ stroke: '#ec9b87', strokeWidth: 2, strokeDasharray: '5 4' }} shape="diamond" isAnimationActive={false} /> : null}
          <Scatter name={primaryName} data={primary} fill="#83e7d4" line={{ stroke: '#83e7d4', strokeWidth: 2.5 }} shape="circle" isAnimationActive={false} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
    <div className="chart-legend"><span><i />{primaryName}</span>{showReference || secondScenario ? <span><i className="coral" />{secondaryName}</span> : null}</div>
    <p className="depth-caption">Depth increases downward <span aria-hidden="true">↓</span> <span className="sr-only">At {selectedDepth} m: {formatValue(active?.estimate)} degrees Celsius.</span></p>
  </div>;
}
