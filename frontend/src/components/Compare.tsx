import { useState } from 'react';
import { ArrowLeftRight, GitCompareArrows, Info } from 'lucide-react';
import { DEPTHS, SCENARIOS } from '../data/scenarios';
import { displayDate, formatValue } from '../lib/calculations';
import { ProfileChart } from './ProfileChart';
import { SurfaceInputs } from './SurfaceInputs';

export function Compare({ firstId, setFirstId }: { firstId: string; setFirstId: (id: string) => void }) {
  const [secondId, setSecondId] = useState('central-post');
  const [depth, setDepth] = useState(100);
  const first = SCENARIOS.find((s) => s.id === firstId);
  const second = SCENARIOS.find((s) => s.id === secondId);
  const identical = firstId === secondId;
  const pointA = first?.profile.find((p) => p.depth === depth);
  const pointB = second?.profile.find((p) => p.depth === depth);
  const difference = pointA?.estimate != null && pointB?.estimate != null ? pointA.estimate-pointB.estimate : null;

  return <>
    <div className="page-intro"><div><div className="eyebrow intro-eyebrow">OCEAN INTELLIGENCE / COMPARE</div><h1>Two scenarios. A deeper perspective<span>.</span></h1><p>Explore how illustrative profiles change across locations and seasons.</p></div></div>
    <div className="compare-grid">
      <div className="comparison-controls">
        {[{ scenario: first, id: firstId, set: setFirstId, letter: 'A', tone: 'mint' }, { scenario: second, id: secondId, set: setSecondId, letter: 'B', tone: 'coral' }].map(({ scenario, id, set, letter, tone }) => <section className={`panel compare-scenario ${tone}`} key={letter}><div className="comparison-heading"><span className="letter-badge">{letter}</span><label htmlFor={`scenario-${letter}`}>Scenario {letter}</label><span className="synthetic-tag">Synthetic</span></div><select id={`scenario-${letter}`} value={id} onChange={(event) => set(event.target.value)}>{SCENARIOS.map((s) => <option key={s.id} value={s.id}>{s.location.name} · {s.season}</option>)}</select>{scenario ? <><p className="compare-coordinate">{displayDate(scenario.date)} <span>·</span> {scenario.location.latitude.toFixed(1)}° N, {scenario.location.longitude.toFixed(1)}° E</p><SurfaceInputs inputs={scenario.inputs} compact /><p className="compare-description">{scenario.description}</p></> : <p role="status">This scenario is unavailable. Select a prepared example.</p>}</section>)}
        <button className="button secondary swap-button" onClick={() => { setFirstId(secondId); setSecondId(firstId); }}><ArrowLeftRight size={16} />Swap scenarios</button>
        <div className="demo-note"><Info size={18} /><p>These are prepared synthetic profiles. Differences illustrate the experience, not measured seasonal change.</p></div>
      </div>
      <section className="panel compare-chart-panel"><div className="panel-heading"><div><div className="eyebrow">SIDE BY SIDE, AT EVERY DEPTH</div><h2>Profile comparison</h2></div><GitCompareArrows size={20} className="text-mint" /></div>{first && second ? <>{identical ? <div className="inline-notice" role="status"><Info size={16} />Both selections are the same. Choose a different scenario to explore a difference.</div> : null}<ProfileChart scenario={first} secondScenario={second} selectedDepth={depth} /><div className="comparison-depth"><label htmlFor="comparison-depth">Compare at depth</label><select id="comparison-depth" value={depth} onChange={(event) => setDepth(Number(event.target.value))}>{DEPTHS.map((d) => <option key={d} value={d}>{d} metres</option>)}</select></div><div className="comparison-readings"><div><span><i className="legend-dot" />Scenario A</span><strong>{formatValue(pointA?.estimate)}<small> °C</small></strong></div><div><span><i className="legend-dot coral" />Scenario B</span><strong>{formatValue(pointB?.estimate)}<small> °C</small></strong></div><div><span>A − B at {depth} m</span><strong>{difference != null && difference > 0 ? '+' : ''}{formatValue(difference)}<small> °C</small></strong></div></div><div className="table-scroll comparison-table"><table><caption>Illustrative estimates at every prepared depth.</caption><thead><tr><th scope="col">Depth (m)</th><th scope="col">A (°C)</th><th scope="col">B (°C)</th><th scope="col">A − B (°C)</th></tr></thead><tbody>{DEPTHS.map((d) => { const a = first.profile.find((p) => p.depth === d)?.estimate; const b = second.profile.find((p) => p.depth === d)?.estimate; return <tr key={d} className={d === depth ? 'selected-row' : ''}><th scope="row">{d}</th><td>{formatValue(a)}</td><td>{formatValue(b)}</td><td>{formatValue(a != null && b != null ? a-b : null)}</td></tr>; })}</tbody></table></div></> : <div className="empty-profile"><h3>Choose two scenarios to compare</h3><p>Prepared examples will appear here once both selections are available.</p></div>}<div className="result-disclaimer">Illustrative output — no trained model is connected</div></section>
    </div>
  </>;
}
