import { CartesianGrid, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts';
import type { Scenario } from '../types';
import { formatValue } from '../lib/calculations';

export type VariableType = 'temperature' | 'soundSpeed' | 'salinity' | 'density';

interface Props {
  scenario: Scenario;
  secondScenario?: Scenario;
  showReference?: boolean;
  selectedDepth?: number;
  variable?: VariableType;
}

export function ProfileChart({
  scenario,
  secondScenario,
  showReference = true,
  selectedDepth = 100,
  variable = 'temperature'
}: Props) {
  const getVal = (p: any) => {
    if (variable === 'soundSpeed') return p.soundSpeed ?? 1500;
    if (variable === 'salinity') return p.salinity ?? 34;
    if (variable === 'density') return p.density ?? 1025;
    return p.estimate;
  };

  const getRefVal = (p: any) => {
    if (variable === 'soundSpeed') return p.soundSpeed != null ? p.soundSpeed + 1.2 : 1500;
    if (variable === 'salinity') return p.salinity != null ? p.salinity + 0.1 : 34;
    if (variable === 'density') return p.density != null ? p.density + 0.2 : 1025;
    return p.reference;
  };

  const unit = variable === 'temperature' ? '°C' : variable === 'soundSpeed' ? ' m/s' : variable === 'salinity' ? ' PSU' : ' kg/m³';
  const varLabel = variable === 'temperature' ? 'TEMPERATURE' : variable === 'soundSpeed' ? 'SOUND SPEED (SVP)' : variable === 'salinity' ? 'SALINITY' : 'SEAWATER DENSITY';

  const xDomain = variable === 'temperature' ? [0, 35] : variable === 'soundSpeed' ? [1470, 1550] : variable === 'salinity' ? [31, 37] : [1018, 1035];
  const maxDepth = Math.max(...scenario.profile.map(p => p.depth), 500);

  const primary = scenario.profile.map((p) => ({ depth: p.depth, val: getVal(p) }));
  const secondary = secondScenario
    ? secondScenario.profile.map((p) => ({ depth: p.depth, val: getVal(p) }))
    : scenario.profile.map((p) => ({ depth: p.depth, val: getRefVal(p) }));

  const isLive = scenario.isLiveModel;
  const primaryName = secondScenario
    ? `${scenario.location.name} · ${scenario.season}`
    : isLive ? 'AI Model Estimate (v2 CNN)' : 'Demo estimate';
  const secondaryName = secondScenario
    ? `${secondScenario.location.name} · ${secondScenario.season}`
    : 'Reference ground truth';

  const active = scenario.profile.find((p) => p.depth === selectedDepth);

  return (
    <div className="profile-visual">
      <div className="chart-axis-title">{varLabel} ({unit.trim()})</div>
      <div
        className="profile-chart"
        role="img"
        aria-label={`${varLabel} versus depth for ${scenario.location.name}. Depth increases downwards from 0 to ${maxDepth} metres.`}
      >
        <ResponsiveContainer width="100%" height="100%" minWidth={0}>
          <ScatterChart margin={{ top: 16, right: 17, bottom: 7, left: 3 }}>
            <CartesianGrid stroke="#263741" strokeDasharray="3 5" horizontal vertical />
            <XAxis
              type="number"
              dataKey="val"
              name={varLabel}
              unit={unit}
              domain={xDomain}
              orientation="top"
              tick={{ fill: '#a2b5c0', fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              tickMargin={9}
            />
            <YAxis
              type="number"
              dataKey="depth"
              name="Depth"
              unit=" m"
              domain={[0, maxDepth]}
              reversed
              tick={{ fill: '#a2b5c0', fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              width={53}
            />
            <ReferenceLine y={selectedDepth} stroke="#6bbcaf" strokeDasharray="4 4" strokeOpacity={0.7} />
            {scenario.thermoclineDepth && variable === 'temperature' ? (
              <ReferenceLine y={scenario.thermoclineDepth} stroke="#eab308" strokeDasharray="2 2" strokeOpacity={0.8} label={{ value: `Thermocline ~${scenario.thermoclineDepth}m`, fill: '#eab308', fontSize: 10, position: 'right' }} />
            ) : null}
            {scenario.sofarAxisDepth && variable === 'soundSpeed' ? (
              <ReferenceLine y={scenario.sofarAxisDepth} stroke="#38bdf8" strokeDasharray="2 2" strokeOpacity={0.8} label={{ value: `SOFAR Axis ~${scenario.sofarAxisDepth}m`, fill: '#38bdf8', fontSize: 10, position: 'right' }} />
            ) : null}
            <Tooltip
              cursor={{ stroke: '#638c9c', strokeDasharray: '3 3' }}
              content={({ active: hovered, payload }) => {
                if (!hovered || !payload?.length) return null;
                const depth = payload[0]?.payload?.depth as number | undefined;
                const point = scenario.profile.find((p) => p.depth === depth);
                if (!point) return null;
                const v = getVal(point);
                const r = getRefVal(point);
                const delta = v != null && r != null ? Math.abs(v - r) : null;
                return (
                  <div className="chart-tooltip">
                    <strong>{depth} m below surface</strong>
                    <span>{primaryName}: {formatValue(v)}{unit}</span>
                    {showReference || secondScenario ? (
                      <>
                        <span>{secondaryName}: {formatValue(r)}{unit}</span>
                        <span>Diff: {formatValue(delta, 2)}{unit}</span>
                      </>
                    ) : null}
                  </div>
                );
              }}
            />
            {showReference || secondScenario ? (
              <Scatter
                name={secondaryName}
                data={secondary}
                fill="#ec9b87"
                line={{ stroke: '#ec9b87', strokeWidth: 2, strokeDasharray: '5 4' }}
                shape="diamond"
                isAnimationActive={false}
              />
            ) : null}
            <Scatter
              name={primaryName}
              data={primary}
              fill="#83e7d4"
              line={{ stroke: '#83e7d4', strokeWidth: 2.5 }}
              shape="circle"
              isAnimationActive={false}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <div className="chart-legend">
        <span><i />{primaryName}</span>
        {showReference || secondScenario ? <span><i className="coral" />{secondaryName}</span> : null}
      </div>
      <p className="depth-caption">
        Depth increases downward <span aria-hidden="true">↓</span>
        <span className="sr-only">At {selectedDepth} m: {formatValue(getVal(active))} {unit}.</span>
      </p>
    </div>
  );
}
