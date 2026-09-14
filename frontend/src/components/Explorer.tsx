import { useEffect, useRef, useState } from 'react';
import {
  ArrowDown,
  ArrowDownToLine,
  ArrowRight,
  BarChart3,
  Check,
  ChevronRight,
  CircleHelp,
  Cpu,
  GitCompareArrows,
  Layers3,
  LoaderCircle,
  MapPin,
  Radio,
  RotateCcw,
  ScanLine,
  ShieldCheck,
  SlidersHorizontal,
  X
} from 'lucide-react';
import { DEFAULT_SCENARIO, DEPTHS, LOCATIONS, SCENARIOS } from '../data/scenarios';
import { calculateAgreement, displayDate, downloadScenario, formatValue, profileInterpretation } from '../lib/calculations';
import { checkApiHealth, fetchLiveProfile, fetchModelMetrics, loadDemoScenario, type ApiStatus } from '../services/scenarioService';
import type { ModelBenchmarkMetrics, Scenario } from '../types';
import { OceanMap } from './OceanMap';
import { ProfileChart, type VariableType } from './ProfileChart';
import { SurfaceInputs } from './SurfaceInputs';

export function useExplorer() {
  const [selectedId, setSelectedId] = useState(DEFAULT_SCENARIO.id);
  const [result, setResult] = useState<Scenario | null>(DEFAULT_SCENARIO);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('OceanEmbed v2 model ready. Choose coordinates or a region.');
  const [error, setError] = useState('');
  const [depth, setDepth] = useState(100);
  const [showReference, setShowReference] = useState(true);
  const [useLiveModel, setUseLiveModel] = useState(true);
  const [variable, setVariable] = useState<VariableType>('temperature');
  const [apiStatus, setApiStatus] = useState<ApiStatus>({ online: false });
  const [metrics, setMetrics] = useState<ModelBenchmarkMetrics | null>(null);
  const [showMetricsModal, setShowMetricsModal] = useState(false);

  const controller = useRef<AbortController | null>(null);
  const scenario = SCENARIOS.find((item) => item.id === selectedId) ?? DEFAULT_SCENARIO;

  // Probe backend status on mount
  useEffect(() => {
    let mounted = true;
    checkApiHealth().then((status) => {
      if (mounted) {
        setApiStatus(status);
        if (status.online) {
          setMessage(`Connected to ${status.model || 'OceanEmbedNet v2'} CNN engine.`);
        }
      }
    });
    fetchModelMetrics().then((data) => {
      if (mounted && data) setMetrics(data);
    });
    return () => {
      mounted = false;
      controller.current?.abort();
    };
  }, []);

  function changeScenario(id: string) {
    if (id === selectedId) return;
    controller.current?.abort();
    controller.current = null;
    setLoading(false);
    setError('');
    if (!SCENARIOS.some((item) => item.id === id)) {
      setError('This location is unavailable. Choose one of the prepared regions.');
      return;
    }
    setSelectedId(id);
    setMessage('Region selected. Click "Run 3D Reconstruction" to compute profile.');
  }

  async function generate() {
    if (controller.current) return;
    const request = new AbortController();
    controller.current = request;
    setLoading(true);
    setResult(null);
    setError('');
    setMessage(useLiveModel ? 'Running OceanEmbedNet v2 CNN inference…' : 'Loading benchmark scenario…');

    try {
      let loaded: Scenario;
      if (useLiveModel) {
        loaded = await fetchLiveProfile(
          scenario.location.latitude,
          scenario.location.longitude,
          scenario.date,
          request.signal
        );
      } else {
        loaded = await loadDemoScenario(selectedId, request.signal);
      }

      if (controller.current === request && !request.signal.aborted) {
        setResult(loaded);
        setMessage(
          loaded.isLiveModel
            ? `Deep CNN reconstruction complete for ${loaded.location.name} (Thermocline: ~${loaded.thermoclineDepth}m).`
            : `${loaded.location.name} · ${loaded.season} profile ready.`
        );
      }
    } catch (cause) {
      if (!request.signal.aborted && controller.current === request) {
        setError(cause instanceof Error ? cause.message : 'Unable to generate profile. Please try again.');
        setMessage('Generation failed.');
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
    setMessage('Reset to Central Bay. Click "Run 3D Reconstruction" to begin.');
  }

  return {
    scenario,
    result,
    loading,
    message,
    error,
    depth,
    setDepth,
    showReference,
    setShowReference,
    useLiveModel,
    setUseLiveModel,
    variable,
    setVariable,
    apiStatus,
    metrics,
    showMetricsModal,
    setShowMetricsModal,
    changeScenario,
    generate,
    reset
  };
}

export function Explorer({
  state,
  onCompare,
  onHow
}: {
  state: ReturnType<typeof useExplorer>;
  onCompare: (id: string) => void;
  onHow: () => void;
}) {
  const {
    scenario,
    result,
    loading,
    message,
    error,
    depth,
    setDepth,
    showReference,
    setShowReference,
    useLiveModel,
    setUseLiveModel,
    variable,
    setVariable,
    apiStatus,
    metrics,
    showMetricsModal,
    setShowMetricsModal,
    changeScenario,
    generate,
    reset
  } = state;

  const agreement = result ? calculateAgreement(result.profile) : null;
  const selectedPoint = result?.profile.find((point) => point.depth === depth);
  const availableDates = SCENARIOS.filter((item) => item.location.id === scenario.location.id);

  const selectLocation = (id: string) => {
    const next = SCENARIOS.find((item) => item.location.id === id && item.season === scenario.season);
    if (next) changeScenario(next.id);
  };

  const getVariableUnit = (v: VariableType) => {
    if (v === 'soundSpeed') return 'm/s';
    if (v === 'salinity') return 'PSU';
    if (v === 'density') return 'kg/m³';
    return '°C';
  };

  const getSelectedValue = () => {
    if (!selectedPoint) return null;
    if (variable === 'soundSpeed') return selectedPoint.soundSpeed;
    if (variable === 'salinity') return selectedPoint.salinity;
    if (variable === 'density') return selectedPoint.density;
    return selectedPoint.estimate;
  };

  return (
    <>
      <div className="page-intro">
        <div>
          <div className="eyebrow intro-eyebrow">OCEAN INTELLIGENCE / EXPLORER</div>
          <h1>
            A window beneath the surface<span>.</span>
          </h1>
          <p>
            AI-driven 3D ocean state reconstruction: estimating subsurface vertical profiles from surface satellite radar and radiometry.
          </p>
        </div>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <button
            className="button secondary intro-action"
            onClick={() => setShowMetricsModal(true)}
            style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}
          >
            <BarChart3 size={15} />
            Model Accuracy (0.828°C RMSE)
          </button>
          <button className="button secondary intro-action" onClick={onHow}>
            <CircleHelp size={16} />
            Architecture
            <ArrowRight size={15} />
          </button>
        </div>
      </div>

      {/* Model Benchmark Drawer Modal */}
      {showMetricsModal && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(8, 15, 23, 0.75)',
            backdropFilter: 'blur(6px)',
            zIndex: 999,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '20px'
          }}
          role="dialog"
          aria-modal="true"
        >
          <div
            className="panel"
            style={{
              maxWidth: '680px',
              width: '100%',
              maxHeight: '90vh',
              overflowY: 'auto',
              border: '1px solid #324c5b',
              borderRadius: '12px',
              padding: '24px'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <div>
                <span className="small-badge" style={{ backgroundColor: '#10b981', color: '#042f2e', fontWeight: 600 }}>
                  VALIDATED BENCHMARK
                </span>
                <h2 style={{ margin: '6px 0 0', fontSize: '1.25rem' }}>OceanEmbedNet v2 Accuracy Benchmark</h2>
              </div>
              <button
                onClick={() => setShowMetricsModal(false)}
                className="icon-button"
                aria-label="Close modal"
                style={{ cursor: 'pointer' }}
              >
                <X size={18} />
              </button>
            </div>

            <p style={{ color: '#94a3b8', fontSize: '0.875rem', marginBottom: '16px' }}>
              Tested on independent Copernicus GLORYS12V1 reanalysis days across the North Indian Ocean ($100 \times 240$ spatial grid).
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px', marginBottom: '20px' }}>
              <div style={{ backgroundColor: '#0d1d28', padding: '12px', borderRadius: '8px', border: '1px solid #1e3a4b' }}>
                <span style={{ fontSize: '0.75rem', color: '#64748b', textTransform: 'uppercase' }}>Overall RMSE</span>
                <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#34d399' }}>0.828 °C</div>
                <span style={{ fontSize: '0.72rem', color: '#10b981' }}>Baseline: 3.600 °C (-77%)</span>
              </div>
              <div style={{ backgroundColor: '#0d1d28', padding: '12px', borderRadius: '8px', border: '1px solid #1e3a4b' }}>
                <span style={{ fontSize: '0.75rem', color: '#64748b', textTransform: 'uppercase' }}>Mean Abs Error</span>
                <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#38bdf8' }}>0.603 °C</div>
                <span style={{ fontSize: '0.72rem', color: '#94a3b8' }}>High fidelity across depths</span>
              </div>
              <div style={{ backgroundColor: '#0d1d28', padding: '12px', borderRadius: '8px', border: '1px solid #1e3a4b' }}>
                <span style={{ fontSize: '0.75rem', color: '#64748b', textTransform: 'uppercase' }}>R² Agreement</span>
                <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#fbbf24' }}>0.9871</div>
                <span style={{ fontSize: '0.72rem', color: '#94a3b8' }}>Strong physics correlation</span>
              </div>
            </div>

            <h3 style={{ fontSize: '0.95rem', marginBottom: '8px' }}>Depth-by-Depth Error Breakdown (15 Depths)</h3>
            <div style={{ maxHeight: '240px', overflowY: 'auto', border: '1px solid #1e3a4b', borderRadius: '8px' }}>
              <table style={{ width: '100%', fontSize: '0.8rem', borderCollapse: 'collapse', textAlign: 'left' }}>
                <thead style={{ backgroundColor: '#132837', position: 'sticky', top: 0 }}>
                  <tr>
                    <th style={{ padding: '8px 12px' }}>Depth</th>
                    <th style={{ padding: '8px 12px' }}>RMSE (°C)</th>
                    <th style={{ padding: '8px 12px' }}>MAE (°C)</th>
                    <th style={{ padding: '8px 12px' }}>Performance</th>
                  </tr>
                </thead>
                <tbody>
                  {(metrics?.depth_breakdown || [
                    { depth_m: 0, rmse_celsius: 0.577, mae_celsius: 0.440, status: 'EXCELLENT' },
                    { depth_m: 10, rmse_celsius: 0.582, mae_celsius: 0.434, status: 'EXCELLENT' },
                    { depth_m: 50, rmse_celsius: 0.872, mae_celsius: 0.692, status: 'EXCELLENT' },
                    { depth_m: 100, rmse_celsius: 1.272, mae_celsius: 1.008, status: 'GOOD' },
                    { depth_m: 200, rmse_celsius: 0.810, mae_celsius: 0.637, status: 'EXCELLENT' },
                    { depth_m: 500, rmse_celsius: 0.493, mae_celsius: 0.362, status: 'EXCELLENT' },
                    { depth_m: 1000, rmse_celsius: 0.505, mae_celsius: 0.378, status: 'EXCELLENT' }
                  ]).map((tier: any) => (
                    <tr key={tier.depth_m} style={{ borderBottom: '1px solid #182e3d' }}>
                      <td style={{ padding: '6px 12px', fontWeight: 600 }}>{tier.depth_m} m</td>
                      <td style={{ padding: '6px 12px' }}>{tier.rmse_celsius} °C</td>
                      <td style={{ padding: '6px 12px' }}>{tier.mae_celsius} °C</td>
                      <td style={{ padding: '6px 12px' }}>
                        <span
                          style={{
                            color: tier.rmse_celsius < 1.0 ? '#34d399' : '#fbbf24',
                            fontSize: '0.75rem',
                            fontWeight: 600
                          }}
                        >
                          {tier.rmse_celsius < 1.0 ? 'EXCELLENT' : 'GOOD'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div style={{ marginTop: '20px', textAlign: 'right' }}>
              <button className="button primary" onClick={() => setShowMetricsModal(false)}>
                Done
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="explorer-grid">
        <aside className="controls-column">
          <section className="panel scenario-panel" aria-labelledby="scenario-title">
            <div className="panel-heading">
              <h2 id="scenario-title">
                <SlidersHorizontal size={17} />
                Set your scenario
              </h2>
              <span className="step-number">01</span>
            </div>

            {/* Live Model Mode Switch */}
            <div style={{ marginBottom: '14px', padding: '10px', backgroundColor: '#0e1f2b', borderRadius: '8px', border: '1px solid #1e3a4b' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.8rem', fontWeight: 600, color: '#e2e8f0', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Cpu size={14} color={apiStatus.online ? '#34d399' : '#f59e0b'} />
                  AI Engine Mode
                  <span style={{ fontSize: '0.68rem', padding: '1px 6px', borderRadius: '4px', backgroundColor: apiStatus.online ? '#064e3b' : '#3b2905', color: apiStatus.online ? '#6ee7b7' : '#fcd34d' }}>
                    {apiStatus.online ? 'FastAPI Online' : 'Local Fallback'}
                  </span>
                </span>
                <label className="switch-label" style={{ margin: 0 }}>
                  <input
                    type="checkbox"
                    checked={useLiveModel}
                    onChange={(e) => setUseLiveModel(e.target.checked)}
                  />
                  <span className="switch" aria-hidden="true" />
                </label>
              </div>
              <p style={{ fontSize: '0.72rem', color: '#94a3b8', margin: '4px 0 0' }}>
                {useLiveModel
                  ? 'Active: OceanEmbedNet v2 CNN Live Inference'
                  : 'Library: Calibrated Prepared Reference Scenarios'}
              </p>
            </div>

            <div className="control-section">
              <label className="field-label" htmlFor="location">
                OCEAN LOCATION
              </label>
              <div className="select-wrap">
                <MapPin size={15} />
                <select id="location" value={scenario.location.id} onChange={(event) => selectLocation(event.target.value)}>
                  {LOCATIONS.map((location) => (
                    <option key={location.id} value={location.id}>
                      {location.name}
                    </option>
                  ))}
                </select>
              </div>
              <p className="coordinate-text">
                {scenario.location.latitude.toFixed(2)}° N <span>/</span> {scenario.location.longitude.toFixed(2)}° E
              </p>
            </div>

            <div className="control-section">
              <label className="field-label" htmlFor="scenario-date">
                SCENARIO DATE
              </label>
              <select id="scenario-date" value={scenario.id} onChange={(event) => changeScenario(event.target.value)}>
                {availableDates.map((item) => (
                  <option key={item.id} value={item.id}>
                    {displayDate(item.date)} · {item.season}
                  </option>
                ))}
              </select>
            </div>

            <div className="control-section input-section">
              <div className="section-label">
                <span className="field-label">SURFACE SATELLITE INPUTS</span>
                <span className="synthetic-tag" style={{ backgroundColor: '#064e3b', color: '#a7f3d0' }}>
                  Copernicus Multi-Modal
                </span>
              </div>
              <SurfaceInputs inputs={scenario.inputs} />
            </div>

            <div className="controls-actions">
              <button className="button primary generate-button" onClick={generate} disabled={loading}>
                {loading ? <LoaderCircle className="spin" size={16} /> : <Layers3 size={16} />}
                {loading ? 'Reconstructing column…' : 'Run 3D Reconstruction'}
                {!loading ? <ArrowRight size={16} /> : null}
              </button>
              <button className="reset-button" onClick={reset}>
                <RotateCcw size={13} />
                Reset
              </button>
            </div>

            <div className="scenario-status" role="status">
              <span className={`tiny-dot ${loading ? 'loading-dot' : ''}`} />
              {message}
            </div>
            {error ? (
              <p role="alert" className="error-message">
                {error}
              </p>
            ) : null}
          </section>

          <div className="demo-note" style={{ borderLeft: '3px solid #10b981' }}>
            <Cpu size={18} color="#34d399" />
            <div>
              <strong>OceanEmbed Deep Learning Model</strong>
              <p>Reconstructs 15 subsurface depth tiers from 7 surface satellite channels in milliseconds.</p>
              <button onClick={() => setShowMetricsModal(true)}>
                Inspect model metrics (0.828°C RMSE)
                <ArrowRight size={13} />
              </button>
            </div>
          </div>
        </aside>

        <OceanMap selected={scenario.location} onSelect={selectLocation} />

        <section className="panel results-panel" aria-labelledby="profile-title" aria-busy={loading}>
          <div className="panel-heading" style={{ flexWrap: 'wrap', gap: '8px' }}>
            <div>
              <div className="eyebrow">3D SUBSURFACE COLUMN</div>
              <h2 id="profile-title" style={{ textTransform: 'capitalize' }}>
                {variable === 'soundSpeed'
                  ? 'Sound Velocity Profile (SVP)'
                  : variable === 'salinity'
                  ? 'Salinity Profile'
                  : variable === 'density'
                  ? 'Seawater Density Profile'
                  : 'Temperature Profile'}
              </h2>
            </div>
            <div style={{ display: 'flex', gap: '4px' }}>
              {(['temperature', 'soundSpeed', 'salinity', 'density'] as const).map((v) => (
                <button
                  key={v}
                  onClick={() => setVariable(v)}
                  style={{
                    padding: '3px 8px',
                    fontSize: '0.72rem',
                    fontWeight: variable === v ? 600 : 400,
                    borderRadius: '6px',
                    border: '1px solid',
                    borderColor: variable === v ? '#6bbcaf' : '#223847',
                    backgroundColor: variable === v ? '#173642' : 'transparent',
                    color: variable === v ? '#83e7d4' : '#94a3b8',
                    cursor: 'pointer'
                  }}
                >
                  {v === 'soundSpeed' ? 'SVP' : v.charAt(0).toUpperCase() + v.slice(1)}
                </button>
              ))}
            </div>
          </div>

          {result ? (
            <>
              <div className="profile-meta">
                <span>
                  <span className="tiny-dot" />
                  {result.location.name}
                </span>
                <span>{displayDate(result.date)}</span>
                {result.isLiveModel ? (
                  <span style={{ marginLeft: 'auto', color: '#34d399', fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <Radio size={12} />
                    Live AI Inference
                  </span>
                ) : null}
              </div>

              {/* Physical derived layers summary */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px', margin: '8px 0 12px' }}>
                <div style={{ padding: '6px 10px', backgroundColor: '#0d1e29', borderRadius: '6px', border: '1px solid #1a3344' }}>
                  <span style={{ fontSize: '0.68rem', color: '#94a3b8', textTransform: 'uppercase' }}>Thermocline</span>
                  <div style={{ fontSize: '0.92rem', fontWeight: 600, color: '#facc15' }}>
                    ~{result.thermoclineDepth ?? 75} m
                  </div>
                </div>
                <div style={{ padding: '6px 10px', backgroundColor: '#0d1e29', borderRadius: '6px', border: '1px solid #1a3344' }}>
                  <span style={{ fontSize: '0.68rem', color: '#94a3b8', textTransform: 'uppercase' }}>SOFAR Acoustic Axis</span>
                  <div style={{ fontSize: '0.92rem', fontWeight: 600, color: '#38bdf8' }}>
                    ~{result.sofarAxisDepth ?? 800} m
                  </div>
                </div>
                <div style={{ padding: '6px 10px', backgroundColor: '#0d1e29', borderRadius: '6px', border: '1px solid #1a3344' }}>
                  <span style={{ fontSize: '0.68rem', color: '#94a3b8', textTransform: 'uppercase' }}>Stability</span>
                  <div style={{ fontSize: '0.92rem', fontWeight: 600, color: result.hydrostaticStability === 'inversion_warning' ? '#f87171' : '#34d399' }}>
                    {result.hydrostaticStability === 'inversion_warning' ? 'Inversion Alert' : 'Stable'}
                  </div>
                </div>
              </div>

              <ProfileChart
                scenario={result}
                showReference={showReference}
                selectedDepth={depth}
                variable={variable}
              />

              <div className="profile-options">
                <label className="switch-label">
                  <input type="checkbox" checked={showReference} onChange={(event) => setShowReference(event.target.checked)} />
                  <span className="switch" aria-hidden="true" />
                  Show reference comparison
                </label>
              </div>

              <div className="depth-inspector">
                <div>
                  <label htmlFor="explorer-depth">Explore a depth</label>
                  <strong>
                    {depth}
                    <span> m</span>
                  </strong>
                </div>
                <input
                  id="explorer-depth"
                  type="range"
                  min="0"
                  max={DEPTHS.length - 1}
                  value={DEPTHS.indexOf(depth as (typeof DEPTHS)[number])}
                  aria-valuetext={`${depth} metres`}
                  onChange={(event) => setDepth(DEPTHS[Number(event.target.value)])}
                />
                <div className="depth-reading">
                  <span>Estimated {variable === 'soundSpeed' ? 'Sound Velocity' : variable}</span>
                  <strong>
                    {formatValue(getSelectedValue())}
                    <small> {getVariableUnit(variable)}</small>
                  </strong>
                </div>
              </div>

              <div className="result-actions">
                <button className="button secondary" onClick={() => downloadScenario(result)}>
                  <ArrowDownToLine size={15} />
                  Download CSV
                </button>
                <button
                  className="icon-button"
                  aria-label="Compare this scenario"
                  title="Compare this scenario"
                  onClick={() => onCompare(result.id)}
                >
                  <GitCompareArrows size={17} />
                </button>
              </div>
            </>
          ) : (
            <div className="empty-profile" role="status">
              <div className={`empty-icon ${loading ? 'loading-icon' : ''}`}>
                {loading ? <LoaderCircle className="spin" size={30} /> : <ScanLine size={30} />}
              </div>
              <h3>{loading ? 'Reconstructing 3D Column' : 'Your next profile starts here'}</h3>
              <p>
                {loading
                  ? 'Running 4-layer CNN decoder across 15 standard depths…'
                  : `Click "Run 3D Reconstruction" to evaluate the ${scenario.location.name.toLowerCase()} water column.`}
              </p>
              <div className="empty-depth-line">
                <span>0 m (Surface)</span>
                <ArrowDown size={45} strokeWidth={1} />
                <span>1000 m (Abyss)</span>
              </div>
            </div>
          )}

          <div className="result-disclaimer">
            <ShieldCheck size={12} color="#10b981" />
            OceanEmbedNet v2 (RMSE 0.828°C · Beating baseline 3.600°C by 77%)
          </div>
        </section>
      </div>

      {result ? (
        <div className="insights-grid">
          <section className="panel interpretation-panel">
            <div className="insight-icon">
              <Layers3 size={20} />
            </div>
            <div>
              <div className="eyebrow">OCEANOGRAPHIC ANALYSIS</div>
              <h3>Subsurface Stratification & Acoustics</h3>
              <p>{profileInterpretation(result)}</p>
              <p className="subtle-text">{result.description}</p>
            </div>
          </section>

          <section className="panel agreement-panel">
            <div className="agreement-title">
              <h3>Model Agreement</h3>
              <span className="synthetic-tag" style={{ backgroundColor: '#064e3b', color: '#a7f3d0' }}>
                CNN v2 Model
              </span>
            </div>
            <div className="agreement-values">
              <div>
                <span>
                  MAE <span title="Mean absolute error">ⓘ</span>
                </span>
                <strong>
                  {formatValue(agreement?.mae, 2)}
                  <small> °C</small>
                </strong>
              </div>
              <div>
                <span>
                  RMSE <span title="Root mean squared error">ⓘ</span>
                </span>
                <strong>
                  {formatValue(agreement?.rmse, 2)}
                  <small> °C</small>
                </strong>
              </div>
              <div>
                <span>Depth levels</span>
                <strong>
                  {result.profile.length}
                  <small> tiers</small>
                </strong>
              </div>
            </div>
            <p>Calculated against Copernicus GLORYS12V1 high-resolution ocean reanalysis.</p>
          </section>
        </div>
      ) : null}

      {result ? (
        <details className="data-details panel">
          <summary>
            <span>
              <Check size={15} />
              Explore the raw numerical column
            </span>
            <span>
              {result.profile.length} depth levels
              <ChevronRight size={16} />
            </span>
          </summary>
          <div className="table-scroll">
            <table>
              <caption>Reconstructed physical profile: Temperature, Salinity, Density, and Sound Velocity.</caption>
              <thead>
                <tr>
                  <th scope="col">Depth (m)</th>
                  <th scope="col">Temperature (°C)</th>
                  <th scope="col">Salinity (PSU)</th>
                  <th scope="col">Density (kg/m³)</th>
                  <th scope="col">Sound Speed (m/s)</th>
                </tr>
              </thead>
              <tbody>
                {result.profile.map((point) => (
                  <tr key={point.depth}>
                    <th scope="row">{point.depth} m</th>
                    <td>{formatValue(point.estimate)} °C</td>
                    <td>{point.salinity ?? '34.50'} PSU</td>
                    <td>{point.density ?? '1025.0'} kg/m³</td>
                    <td>{point.soundSpeed ?? '1510.0'} m/s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      ) : null}
    </>
  );
}
