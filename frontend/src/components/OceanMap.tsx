import { Crosshair, MapPin, Navigation } from 'lucide-react';
import { LOCATIONS } from '../data/scenarios';
import type { OceanLocation } from '../types';

// Simplified, hand-authored coastline. Equirectangular projection shared by all layers.
const project = (lon: number, lat: number) => [((lon - 77) / 22) * 620, ((24 - lat) / 20) * 565];
const polygon = (coordinates: number[][]) => coordinates.map(([lon, lat], i) => `${i ? 'L' : 'M'}${project(lon, lat).map((v) => v.toFixed(1)).join(',')}`).join(' ') + ' Z';
const INDIA = polygon([[76,25],[92,25],[92,23],[91.5,22.3],[90.9,22.4],[90.5,21.9],[89.9,22.1],[89.5,21.8],[88.9,21.6],[88.2,21.6],[87.9,21.5],[87.4,21.2],[86.9,20.7],[86.6,20.1],[86,19.9],[85.4,19.6],[84.8,19],[84.2,18.5],[83.6,18],[83.2,17.5],[82.5,17.1],[82.3,16.6],[81.7,16.3],[81.2,16.3],[80.7,15.9],[80.2,15.7],[80.1,15],[80.3,13.5],[80.2,12.4],[79.9,11.8],[79.8,10.4],[79.3,10],[79.2,9.4],[78.8,9.1],[78.1,8.8],[77.5,8.1],[77,8.3],[76,10]]);
const MYANMAR = polygon([[92,25],[100,25],[100,4],[98.7,5.4],[98.3,7.7],[98.6,9.4],[98.5,10.3],[98.2,12],[97.7,13],[97.8,14.5],[97.5,15.8],[97,16.5],[96.4,16.3],[95.7,16],[95.2,15.7],[94.4,16],[94.2,16.8],[94.6,17.6],[94.2,18.4],[93.6,19.2],[93.2,19.8],[92.7,20.4],[92.4,21],[92.1,21.5],[92,22]]);
const SRI_LANKA = polygon([[79.8,9.8],[80.2,9.5],[80.6,8.7],[81.2,8.4],[81.8,7.1],[81.6,6.5],[80.9,5.9],[80.2,6],[79.8,6.8],[79.7,7.6],[79.9,8.4]]);

export function OceanMap({ selected, onSelect }: { selected: OceanLocation; onSelect: (id: string) => void }) {
  return <section className="map-panel panel" aria-labelledby="map-title">
    <div className="panel-heading"><div><div className="eyebrow">STUDY REGION</div><h2 id="map-title">Bay of Bengal</h2></div><span className="small-badge"><span className="tiny-dot" />5 locations</span></div>
    <div className="map-canvas">
      <svg viewBox="0 0 620 565" aria-label="Illustrative Bay of Bengal map. Choose one of five ocean locations." className="ocean-svg">
        <defs>
          <pattern id="map-grain" width="6" height="6" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r=".55" fill="#7ba7b7" opacity=".09" /></pattern>
          <radialGradient id="sea-shading"><stop offset="0" stopColor="#103341" /><stop offset="1" stopColor="#0b202e" /></radialGradient>
          <clipPath id="map-clip"><rect width="620" height="565" /></clipPath>
        </defs>
        <g clipPath="url(#map-clip)">
          <rect width="620" height="565" fill="url(#sea-shading)" />
          <rect width="620" height="565" fill="url(#map-grain)" />
          <g className="map-contours" fill="none" stroke="#3d7584" strokeWidth=".7" opacity=".22">
            <path d="M155 170C230 95 381 65 437 153S513 269 489 365S367 512 172 543" />
            <path d="M167 210C232 132 375 89 419 168S478 281 456 369S351 488 173 516" />
            <path d="M184 245C237 170 357 124 401 195S453 287 421 373S325 464 184 485" />
            <path d="M201 280C252 218 334 164 379 224S427 291 387 365S303 432 201 452" />
            <path d="M230 302C273 253 319 215 355 249S392 305 356 355S292 399 226 417" />
          </g>
          <g stroke="#85a7b5" strokeWidth=".65" opacity=".12" strokeDasharray="3 6">
            {[80,85,90,95].map((lon) => <path key={lon} d={`M${project(lon,24)[0]} 0V565`} />)}
            {[5,10,15,20].map((lat) => <path key={lat} d={`M0 ${project(77,lat)[1]}H620`} />)}
          </g>
          <g fill="#1b303a" stroke="#49606a" strokeWidth="1.2" strokeLinejoin="round">
            <path d={INDIA} /><path d={MYANMAR} /><path d={SRI_LANKA} />
            <path d={polygon([[92.8,13.7],[93,13.3],[92.9,12.8],[93,12.4],[92.8,11.6],[92.6,11.6],[92.7,12.2],[92.7,12.8]])} />
            <path d={polygon([[93.7,8.2],[93.9,7.9],[93.8,7.1],[93.6,7.1],[93.5,7.6]])} />
            <path d={polygon([[93.5,9.4],[93.7,9],[93.5,8.7],[93.3,9]])} />
          </g>
          <g className="map-country" fill="#8ba0a9" fontSize="11" letterSpacing="3">
            <text x="110" y="112">INDIA</text><text x="322" y="35" fontSize="8" letterSpacing="1.5">BANGLADESH</text>
            <text x="499" y="161" transform="rotate(65 499 161)">MYANMAR</text>
            <text x="34" y="537" fontSize="9" letterSpacing="1.5">SRI LANKA</text>
          </g>
          <text x="316" y="242" textAnchor="middle" className="sea-label">BAY OF BENGAL</text>
          <text x="514" y="405" textAnchor="middle" className="minor-sea-label">ANDAMAN SEA</text>
          <g className="map-coordinates">
            {[80,85,90,95].map((lon) => <text key={lon} x={project(lon,24)[0]} y="552" textAnchor="middle">{lon}°E</text>)}
            {[10,15,20].map((lat) => <text key={lat} x="12" y={project(77,lat)[1]-8}>{lat}°N</text>)}
          </g>
          {LOCATIONS.map((location) => {
            const [x,y] = project(location.longitude,location.latitude);
            const active = selected.id === location.id;
            return <g key={location.id} className={`map-marker ${active ? 'active' : ''}`} role="button" tabIndex={0}
              aria-label={`Select ${location.name}, ${location.latitude}°N, ${location.longitude}°E`} aria-pressed={active}
              onClick={() => onSelect(location.id)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelect(location.id); } }} transform={`translate(${x} ${y})`}>
              <title>{location.name} · {location.latitude}°N, {location.longitude}°E</title>
              <circle className="marker-halo" r={active ? 23 : 17} /><circle className="marker-ring" r={active ? 13 : 9} /><circle className="marker-center" r={active ? 5 : 3.5} />
              {active ? <g transform="translate(0 39)"><rect x="-57" y="-11" width="114" height="26" rx="5" /><text textAnchor="middle" y="6">{location.name}</text></g> : <text className="marker-number" x="15" y="4">{location.label}</text>}
            </g>;
          })}
        </g>
      </svg>
      <div className="north-indicator" role="img" aria-label="North is up"><Navigation size={16} /><span>N</span></div>
      <div className="map-scale"><span />200 km approx.</div>
    </div>
    <div className="map-footer"><span><MapPin size={13} />{selected.latitude.toFixed(2)}° N, {selected.longitude.toFixed(2)}° E</span><Crosshair size={16} /></div>
    <p className="map-disclaimer">Illustrative map · Simplified coastlines · Click a location to explore</p>
  </section>;
}
