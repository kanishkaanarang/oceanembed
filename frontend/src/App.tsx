import { useState } from 'react';
import { BookOpen, Compass, FlaskConical, GitCompareArrows, Waves } from 'lucide-react';
import { Explorer, useExplorer } from './components/Explorer';
import { Compare } from './components/Compare';
import { HowItWorks } from './components/HowItWorks';

type View = 'explorer' | 'compare' | 'how';
const NAV_ITEMS = [{ id: 'explorer', label: 'Explorer', icon: Compass }, { id: 'compare', label: 'Compare', icon: GitCompareArrows }, { id: 'how', label: 'How It Works', icon: BookOpen }] as const;

export default function App() {
  const [view, setView] = useState<View>('explorer');
  const [firstComparison, setFirstComparison] = useState('central-pre');
  const explorer = useExplorer();
  return <div className="app-shell min-h-screen">
    <a href="#main-content" className="skip-link">Skip to content</a>
    <header className="app-header"><div className="header-inner"><button className="brand" aria-label="OceanEmbed home" onClick={() => setView('explorer')}><span className="brand-symbol"><Waves size={25} /></span><span className="brand-text">Ocean<span>Embed</span><small>Explore beneath the surface</small></span></button><div className="header-right"><span className="demo-badge"><FlaskConical size={13} />Demo mode<span className="badge-divider">·</span><span>Synthetic data</span></span><span className="prototype-version">RESEARCH PROTOTYPE <span>v0.1</span></span></div></div></header>
    <div className="navigation-bar" role="region" aria-label="Workspace navigation and study area"><div className="navigation-inner"><nav aria-label="Main navigation">{NAV_ITEMS.map(({ id, label, icon: Icon }) => <button key={id} onClick={() => setView(id)} aria-current={view === id ? 'page' : undefined} className={`nav-item ${view === id ? 'selected' : ''}`}><Icon size={16} />{label}</button>)}</nav><div className="region-label"><span className="tiny-dot" />INDIAN OCEAN<span>/</span>BAY OF BENGAL</div></div></div>
    <main id="main-content" tabIndex={-1} className="main-content">{view === 'explorer' ? <Explorer state={explorer} onCompare={(id) => { setFirstComparison(id); setView('compare'); }} onHow={() => setView('how')} /> : view === 'compare' ? <Compare firstId={firstComparison} setFirstId={setFirstComparison} /> : <HowItWorks onExplore={() => setView('explorer')} />}</main>
    <footer className="app-footer"><div><Waves size={16} /><span>OceanEmbed</span><span className="footer-separator">/</span>From surface signals to deeper understanding.</div><span><span className="tiny-dot" />Local demo · No external data connection</span></footer>
  </div>;
}
