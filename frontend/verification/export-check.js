(async () => {
  const original = URL.createObjectURL;
  let capturedBlob;
  URL.createObjectURL = function (blob) { capturedBlob = blob; return original.call(this, blob); };
  try {
    const button = [...document.querySelectorAll('button')].find((element) => element.textContent.trim() === 'Download CSV');
    if (!button) throw new Error('No export button is available');
    button.click();
  } finally { URL.createObjectURL = original; }
  if (!capturedBlob) throw new Error('The export button did not produce a Blob');
  const text = await capturedBlob.text();
  const lines = text.trim().split('\r\n');
  if (lines.length !== 12 || !lines[0].includes('depth_m') || !lines.slice(1).every((line) => line.includes('synthetic; illustrative output; no trained model'))) {
    throw new Error('Export content or provenance is incorrect');
  }
  return { type: capturedBlob.type, bytes: capturedBlob.size, rows: lines.length-1, provenanceOnEveryRow: true, header: lines[0], firstRow: lines[1], lastRow: lines.at(-1) };
})()
