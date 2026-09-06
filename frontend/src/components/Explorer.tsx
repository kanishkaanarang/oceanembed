import { useEffect, useRef, useState } from 'react';
import { ArrowDown, ArrowDownToLine, ArrowRight, Check, ChevronRight, CircleHelp, FlaskConical, GitCompareArrows, Layers3, LoaderCircle, MapPin, RotateCcw, ScanLine, SlidersHorizontal } from 'lucide-react';
import { DEFAULT_SCENARIO, DEPTHS, LOCATIONS, SCENARIOS } from '../data/scenarios';
import { calculateAgreement, displayDate, downloadScenario, formatValue, profileInterpretation } from '../lib/calculations';
import { loadDemoScenario } from '../services/scenarioService';
import type { Scenario } from '../types';
import { OceanMap } from './OceanMap';
import { ProfileChart } from './ProfileChart';
import { SurfaceInputs } from './SurfaceInputs';

export function useExplorer() {
  const [selectedId, setSelectedId] = useState(DEFAULT_SCENARIO.id);
  const [result, setResult] = useState<Scenario | null>(DEFAULT_SCENARIO);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('Prepared example loaded. Choose a location to explore.');
  const [error, setError] = useState('');
  const [depth, setDepth] = useState(100);
  const [showReference, setShowReference] = useState(true);
  const controller = useRef<AbortController | null>(null);
  const scenario = SCENARIOS.find((item) => item.id === selectedId) ?? DEFAULT_SCENARIO;

  useEffect(() => () => controller.current?.abort(), []);

  function changeScenario(id: string) {
    if (id === selectedId) return;
    controller.current?.abort();
    controller.current = null;
    setLoading(false);
    setResult(null);
    setError('');
    if (!SCENARIOS.some((item) => item.id === id)) {
      setError('This scenario is unavailable. Choose one of the prepared examples.');
      return;
    }
    setSelectedId(id);
    setMessage('Selection changed. Generate a demo profile to view this scenario.');
  }

  async function generate() {
    if (controller.current) return;
    const request = new AbortController();
    controller.current = request;
    setLoading(true);
    setResult(null);
    setError('');
    setMessage('Loading prepared scenario…');
    try {
      const loaded = await loadDemoScenario(selectedId, request.signal);
      if (controller.current === request && !request.signal.aborted) {
        setResult(loaded);
        setMessage(`${loaded.location.name} · ${loaded.season} demo profile ready.`);
      }
    } catch (cause) {
      if (!request.signal.aborted && controller.current === request) {
        setError(cause instanceof Error ? cause.message : 'Unable to load this scenario. Please try again.');
        setMessage('Scenario could not be loaded.');
      }
    } finally {
      if (controller.current === request) {
        controller.current = null;
        setLoading(false);
      }
    }
  }

  function reset() {
    controller.current?.abort();
    controller.current = null;
    setSelectedId(DEFAULT_SCENARIO.id);
    setResult(null);
    setLoading(false);
    setDepth(100);
    setShowReference(true);
    setError('');
    setMessage('Reset to Central Bay. Generate a demo profile to begin.');
  }

  return { scenario, result, loading, message, error, depth, setDepth, showReference, setShowReference, changeScenario, generate, reset };
}

export function Explorer({ state, onCompare, onHow }: { state: ReturnType<typeof useExplorer>; onCompare: (id: string) => void; onHow: () => void }) {
  const { scenario, result, loading, message, error, depth, setDepth, showReference, setShowReference, changeScenario, generate, reset } = state;
  const agreement = result ? calculateAgreement(result.profile) : null;
  const selectedPoint = result?.profile.find((point) => point.depth === depth);
  const availableDates = SCENARIOS.filter((item) => item.location.id === scenario.location.id);

  const selectLocation = (id: string) => {
    const next = SCENARIOS.find((item) => item.location.id === id && item.season === scenario.season);
    if (next) changeScenario(next.id);
  };

  return <>
    <div className="page-intro"><div><div className="eyebrow intro-eyebrow">OCEAN INTELLIGENCE / EXPLORER</div><h1>A window beneath the surface<span>.</span></h1><p>Explore how surface conditions could reveal the ocean below.</p></div><button className="button secondary intro-action" onClick={onHow}><CircleHelp size={16} />About this demo<ArrowRight size={15} /></button></div>
    <div className="explorer-grid">
      <aside className="controls-column">
        <section className="panel scenario-panel" aria-labelledby="scenario-title">
          <div className="panel-heading"><h2 id="scenario-title"><SlidersHorizontal size={17} />Set your scenario</h2><span className="step-number">01</span></div>
          <div className="control-section"><label className="field-label" htmlFor="location">OCEAN LOCATION</label><div className="select-wrap"><MapPin size={15} /><select id="location" value={scenario.location.id} onChange={(event) => selectLocation(event.target.value)}>{LOCATIONS.map((location) => <option key={location.id} value={location.id}>{location.name}</option>)}</select></div><p className="coordinate-text">{scenario.location.latitude.toFixed(2)}° N <span>/</span> {scenario.location.longitude.toFixed(2)}° E</p></div>
          <div className="control-section"><label className="field-label" htmlFor="scenario-date">SCENARIO DATE</label><select id="scenario-date" value={scenario.id} onChange={(event) => changeScenario(event.target.value)}>{availableDates.map((item) => <option key={item.id} value={item.id}>{displayDate(item.date)} · {item.season}</option>)}</select><p className="field-help">Two prepared seasonal examples per location.</p></div>
          <div className="control-section input-section"><div className="section-label"><span className="field-label">SURFACE INPUTS</span><span className="synthetic-tag">Synthetic</span></div><SurfaceInputs inputs={scenario.inputs} /></div>
          <div className="controls-actions"><button className="button primary generate-button" onClick={generate} disabled={loading}>{loading ? <LoaderCircle className="spin" size={16} /> : <Layers3 size={16} />}{loading ? 'Loading scenario…' : 'Generate demo profile'}{!loading ? <ArrowRight size={16} /> : null}</button><button className="reset-button" onClick={reset}><RotateCcw size={13} />Reset scenario</button></div>
          <div className="scenario-status" role="status"><span className={`tiny-dot ${loading ? 'loading-dot' : ''}`} />{message}</div>
          {error ? <p role="alert" className="error-message">{error}</p> : null}
        </section>
        <div className="demo-note"><FlaskConical size={18} /><div><strong>An idea you can explore.</strong><p>Prepared data demonstrates the workflow. No trained model is connected.</p><button onClick={onHow}>See the research roadmap<ArrowRight size={13} /></button></div></div>
      </aside>
      <OceanMap selected={scenario.location} onSelect={selectLocation} />
      <section className="panel results-panel" aria-labelledby="profile-title" aria-busy={loading}>
        <div className="panel-heading"><div><div className="eyebrow">BELOW THE SURFACE</div><h2 id="profile-title">Temperature profile</h2></div><span className="small-badge">0–500 m</span></div>
        {result ? <>
          <div className="profile-meta"><span><span className="tiny-dot" />{result.location.name}</span><span>{displayDate(result.date)}</span></div>
          <ProfileChart scenario={result} showReference={showReference} selectedDepth={depth} />
          <div className="profile-options"><label className="switch-label"><input type="checkbox" checked={showReference} onChange={(event) => setShowReference(event.target.checked)} /><span className="switch" aria-hidden="true" />Show synthetic reference</label></div>
          <div className="depth-inspector"><div><label htmlFor="explorer-depth">Explore a depth</label><strong>{depth}<span> m</span></strong></div><input id="explorer-depth" type="range" min="0" max={DEPTHS.length-1} value={DEPTHS.indexOf(depth as typeof DEPTHS[number])} aria-valuetext={`${depth} metres`} onChange={(event) => setDepth(DEPTHS[Number(event.target.value)])} /><div className="depth-reading"><span>Illustrative temperature</span><strong>{formatValue(selectedPoint?.estimate)}<small> °C</small></strong></div></div>
          <div className="result-actions"><button className="button secondary" onClick={() => downloadScenario(result)}><ArrowDownToLine size={15} />Download CSV</button><button className="icon-button" aria-label="Compare this scenario" title="Compare this scenario" onClick={() => onCompare(result.id)}><GitCompareArrows size={17} /></button></div>
        </> : <div className="empty-profile" role="status"><div className={`empty-icon ${loading ? 'loading-icon' : ''}`}>{loading ? <LoaderCircle className="spin" size={30} /> : <ScanLine size={30} />}</div><h3>{loading ? 'Loading prepared scenario' : 'Your next profile starts here'}</h3><p>{loading ? 'Reading a local synthetic example. No model inference is running.' : `Generate the ${scenario.location.name.toLowerCase()} demo profile to explore its temperature at depth.`}</p><div className="empty-depth-line"><span>0 m</span><ArrowDown size={45} strokeWidth={1} /><span>500 m</span></div></div>}
        <div className="result-disclaimer"><FlaskConical size={12} />Illustrative output — no trained model is connected</div>
      </section>
    </div>
    {result ? <div className="insights-grid">
      <section className="panel interpretation-panel"><div className="insight-icon"><Layers3 size={20} /></div><div><div className="eyebrow">READING THIS PROFILE</div><h3>From a warm surface to cooler depths</h3><p>{profileInterpretation(result)}</p><p className="subtle-text">{result.description}</p></div></section>
      <section className="panel agreement-panel"><div className="agreement-title"><h3>Scenario agreement</h3><span className="synthetic-tag">Synthetic pairs</span></div><div className="agreement-values"><div><span>MAE <span title="Mean absolute error">ⓘ</span></span><strong>{formatValue(agreement?.mae,2)}<small> °C</small></strong></div><div><span>RMSE <span title="Root mean squared error">ⓘ</span></span><strong>{formatValue(agreement?.rmse,2)}<small> °C</small></strong></div><div><span>Depth pairs</span><strong>{agreement?.count ?? 0}<small> levels</small></strong></div></div><p>Calculated from synthetic pairs; this is not model validation.</p></section>
    </div> : null}
    {result ? <details className="data-details panel"><summary><span><Check size={15} />Explore the numbers</span><span>11 depth levels<ChevronRight size={16} /></span></summary><div className="table-scroll"><table><caption>All values are synthetic. Temperature in °C; depth in metres.</caption><thead><tr><th scope="col">Depth (m)</th><th scope="col">Illustrative estimate (°C)</th><th scope="col">Synthetic reference (°C)</th><th scope="col">Absolute difference (°C)</th></tr></thead><tbody>{result.profile.map((point) => <tr key={point.depth}><th scope="row">{point.depth}</th><td>{formatValue(point.estimate)}</td><td>{formatValue(point.reference)}</td><td>{formatValue(point.estimate != null && point.reference != null ? Math.abs(point.estimate-point.reference) : null,2)}</td></tr>)}</tbody></table></div></details> : null}
  </>;
}
