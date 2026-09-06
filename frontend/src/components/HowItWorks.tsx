import { ArrowRight, ChartNoAxesCombined, Check, CircleDashed, Database, FlaskConical, Layers3, Satellite, ScanLine, Waves } from 'lucide-react';

const WORKFLOW = [
  { icon: Satellite, title: 'Surface observations', caption: 'The visible ocean', text: 'Collect surface temperature, sea-level and salinity products. Record whether each source is satellite-derived, in situ, or modelled.', state: 'Initial acquisition' },
  { icon: Database, title: 'Match & check', caption: 'Connect the evidence', text: 'Match surface inputs to Argo profile locations and times. Check quality, units, missing values and depth coverage.', state: 'Planned' },
  { icon: Layers3, title: 'Train a model', caption: 'Learn the relationship', text: 'Learn a mapping from surface conditions to subsurface temperatures, with a simple baseline and carefully separated test data.', state: 'Planned' },
  { icon: ScanLine, title: 'Estimate a profile', caption: 'Look below the surface', text: 'Produce a temperature estimate at each depth. This demo illustrates the output using prepared synthetic profiles.', state: 'Demo visualization' },
  { icon: ChartNoAxesCombined, title: 'Validate with Argo', caption: 'Test against measurements', text: 'Evaluate on held-out profiles, report errors by depth, compare with baselines and examine uncertainty.', state: 'Planned' },
];

export function HowItWorks({ onExplore }: { onExplore: () => void }) {
  return <>
    <div className="page-intro"><div><div className="eyebrow intro-eyebrow">OCEAN INTELLIGENCE / THE IDEA</div><h1>The surface is only the beginning<span>.</span></h1><p>A transparent look at the concept, the prototype, and the work ahead.</p></div><button className="button primary intro-action" onClick={onExplore}>Explore the demo<ArrowRight size={16} /></button></div>
    <section className="concept-banner panel"><div className="concept-art" aria-hidden="true"><Waves size={58} strokeWidth={1} /><div /><span>0 m</span><span>500 m</span></div><div><div className="eyebrow">WHY OCEANEMBED?</div><h2>Make the hidden ocean easier to explore.</h2><p>Surface observations cover broad areas, while subsurface measurements are sparse. OceanEmbed explores whether their relationship can help estimate temperature below the surface, with a focus on the Bay of Bengal.</p><p>This concept is designed for an ocean researcher exploring temperature structure. Its usefulness must still be demonstrated through model development and evaluation.</p></div></section>
    <div className="section-heading"><div><div className="eyebrow">FROM INPUT TO EVIDENCE</div><h2>The proposed research workflow</h2></div><span className="small-badge">5 connected stages</span></div>
    <div className="workflow-grid">{WORKFLOW.map(({ icon: Icon, title, caption, text, state }, i) => <section className="panel workflow-card" key={title}><div className="workflow-top"><Icon size={24} /><span>0{i+1}</span></div><p className="workflow-caption">{caption}</p><h3>{title}</h3><p>{text}</p><span className={`workflow-state ${state === 'Planned' ? 'planned' : ''}`}>{state === 'Planned' ? <CircleDashed size={12} /> : <Check size={12} />}{state}</span>{i < WORKFLOW.length-1 ? <ArrowRight className="workflow-arrow" size={17} /> : null}</section>)}</div>
    <div className="roadmap-grid">
      <section className="panel roadmap-card"><div className="roadmap-icon"><Check size={19} /></div><div className="eyebrow">AVAILABLE NOW</div><h3>In this frontend demo</h3><ul><li>10 deterministic synthetic scenarios</li><li>Interactive map and depth profiles</li><li>Season and location comparison</li><li>Transparent synthetic agreement metrics</li><li>Local CSV export with provenance</li></ul></section>
      <section className="panel roadmap-card"><div className="roadmap-icon blue"><Database size={19} /></div><div className="eyebrow">RESEARCH FOUNDATION</div><h3>In the existing repository</h3><ul><li>Initial Argo data-fetching scripts</li><li>Copernicus temperature acquisition</li><li>Exploratory notebooks and plots</li><li>Near-surface reference comparisons</li></ul><p>Existing Copernicus downloads are model/reanalysis products. Reference comparisons are not AI prediction accuracy.</p></section>
      <section className="panel roadmap-card"><div className="roadmap-icon coral"><CircleDashed size={19} /></div><div className="eyebrow">NEXT MILESTONES</div><h3>From concept to model</h3><ul><li>Quality-checked, matched training data</li><li>Baseline model and saved weights</li><li>Learned embeddings / deep models</li><li>Uncertainty and explainability</li><li>Independent evaluation on held-out data</li></ul></section>
    </div>
    <section className="panel glossary"><div className="panel-heading"><div><div className="eyebrow">A LITTLE OCEAN LITERACY</div><h2>The terms behind the charts</h2></div></div><dl>{[
      ['SST', 'Sea surface temperature', 'How warm the water is near the ocean surface, shown here in degrees Celsius.'],
      ['SLA', 'Sea-level anomaly', 'How far sea level differs from a reference average, shown here in metres.'],
      ['SSS', 'Sea surface salinity', 'The saltiness of surface water, displayed using the conventional PSU label.'],
      ['ARGO', 'Ocean profiling floats', 'Drifting instruments that measure temperature and salinity through the water column. No actual Argo measurements are bundled in this demo.'],
      ['PROFILE', 'Temperature at depth', 'A curve showing temperature at different depths. A thermocline is a layer where temperature changes rapidly with depth.'],
    ].map(([term, title, text]) => <div key={term}><dt><span>{term}</span>{title}</dt><dd>{text}</dd></div>)}</dl></section>
    <div className="honesty-note"><FlaskConical size={22} /><div><strong>A concept demo, with its boundaries visible.</strong><p>All scenario dates, surface inputs, estimates, and reference profiles are illustrative. No trained model, operational forecast, validated accuracy, or uncertainty estimate is provided. Synthetic agreement only describes the prepared pairs.</p></div></div>
  </>;
}
