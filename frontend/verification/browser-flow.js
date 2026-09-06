// Run in the isolated test browser using agent-browser eval --stdin.
// This checks visible DOM and user-facing state; it does not access React internals.
(async () => {
  const results = [];
  const assert = (condition, label) => {
    if (!condition) throw new Error(label);
    results.push(label);
  };
  const waitFor = async (condition, label, timeout = 3000) => {
    const end = Date.now() + timeout;
    while (!condition()) {
      if (Date.now() > end) throw new Error('Timed out: ' + label);
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
  };
  const button = (text) => [...document.querySelectorAll('button')].find((element) => element.textContent.trim() === text);
  const select = async (id, value) => {
    const element = document.getElementById(id);
    if (!element) throw new Error('Missing select: ' + id);
    element.value = value;
    element.dispatchEvent(new Event('change', { bubbles: true }));
    await new Promise((resolve) => setTimeout(resolve, 30));
  };
  const click = async (text) => {
    const element = button(text);
    if (!element) throw new Error('Missing button: ' + text);
    element.click();
    await new Promise((resolve) => setTimeout(resolve, 30));
  };
  const body = () => document.body.innerText;

  await click('Explorer');
  await click('Reset scenario');
  assert(body().includes('Synthetic data') && body().includes('no trained model'), 'Persistent demo provenance');
  assert(!document.querySelector('vite-error-overlay'), 'No Vite error overlay');
  await select('location', 'northern');
  assert(!button('Download CSV') && body().includes('Your next profile starts here'), 'Changing location clears the old result and export');
  await click('Generate demo profile');
  assert(button('Loading scenario…')?.disabled, 'Duplicate generation disabled during loading');
  await select('location', 'western');
  // Wait beyond the fixture delay to ensure a cancelled result never resurfaces.
  await new Promise((resolve) => setTimeout(resolve, 750));
  assert(document.getElementById('location').value === 'western' && !button('Download CSV'), 'Rapid selection cancels stale in-flight results');
  await click('Generate demo profile');
  await waitFor(() => !!button('Download CSV'), 'western profile');
  assert(document.querySelector('.profile-meta').textContent.includes('Western Bay'), 'Generation displays the currently selected location');
  await select('scenario-date', 'western-post');
  assert(!button('Download CSV'), 'Changing date clears prior results');
  await click('Generate demo profile');
  await waitFor(() => !!button('Download CSV'), 'post-monsoon profile');
  assert(document.querySelector('.profile-meta').textContent.includes('15 Nov 2024'), 'Season selection updates result date');

  await waitFor(() => [...document.querySelectorAll('.profile-chart text')].filter((tick) => /^\d+ m$/.test(tick.textContent)).length === 6, 'chart axis labels after layout');
  const ticks = [...document.querySelectorAll('.profile-chart text')].filter((tick) => /^\d+ m$/.test(tick.textContent));
  const positions = ticks.map((tick) => ({ text: tick.textContent, y: tick.getBoundingClientRect().y }));
  assert(positions.length === 6 && positions[0].text.includes('0 m') && positions[5].text.includes('500 m') && positions[0].y < positions[5].y, 'Depth axis runs from surface at top to 500 m at bottom');
  const steps = positions.slice(1).map((point, i) => point.y-positions[i].y);
  assert(Math.max(...steps)-Math.min(...steps) < 2, 'Numeric depth spacing is proportional (within tick-label edge adjustment)');

  const checkbox = document.querySelector('.switch-label input');
  checkbox.click();
  await waitFor(() => !document.querySelector('.chart-legend').textContent.includes('Synthetic reference'), 'reference hidden');
  assert(document.querySelectorAll('.recharts-scatter').length === 1, 'Reference toggle hides its chart series');
  checkbox.click();
  await waitFor(() => document.querySelectorAll('.recharts-scatter').length === 2, 'reference restored');
  assert(true, 'Reference toggle restores its chart series');

  await click('Reset scenario');
  assert(document.getElementById('location').value === 'central' && !button('Download CSV'), 'Reset restores defaults and clears results');
  await click('Generate demo profile');
  await click('Reset scenario');
  await new Promise((resolve) => setTimeout(resolve, 750));
  assert(!button('Download CSV'), 'Reset cancels an in-flight generation');
  await click('Generate demo profile');
  await waitFor(() => !!button('Download CSV'), 'default regeneration');
  document.querySelector('[aria-label="Compare this scenario"]').click();
  await waitFor(() => !!document.getElementById('scenario-A'), 'comparison navigation');
  assert(document.getElementById('scenario-A').value === 'central-pre', 'Compare action carries the current scenario');
  await select('comparison-depth', '100');
  assert(document.querySelector('.comparison-readings').textContent.includes('-1.8'), 'Comparison computes the depth-specific signed difference');
  await click('Swap scenarios');
  assert(document.getElementById('scenario-A').value === 'central-post' && document.getElementById('scenario-B').value === 'central-pre', 'Swap updates both comparison scenarios');
  assert(document.querySelector('.comparison-readings').textContent.includes('+1.8'), 'Swap reverses the signed difference');
  await select('scenario-B', 'central-post');
  assert(body().includes('Both selections are the same') && document.querySelector('.comparison-readings').textContent.includes('0.0'), 'Identical scenario comparison gives an explicit notice and zero difference');
  await click('How It Works');
  assert(body().includes('The proposed research workflow') && body().includes('held-out') && body().includes('In the existing repository'), 'Research workflow and limitations are explained');
  await click('Explore the demo');
  assert(button('Download CSV') && document.getElementById('location').value === 'central', 'Explorer preserves its result across navigation');
  const externalResources = performance.getEntriesByType('resource').map((entry) => entry.name).filter((url) => /^https?:/.test(url) && new URL(url).origin !== location.origin);
  assert(externalResources.length === 0, 'No external runtime resources loaded');
  assert(document.documentElement.scrollWidth <= window.innerWidth, 'No horizontal page overflow');
  return { passed: results.length, checks: results, axis: positions, externalResources };
})()
