() => {
  let drag = null;
  let pan = {x: 0, y: 0};
  function images() { return document.querySelectorAll('#comparison-n img, #comparison-s img'); }
  document.addEventListener('pointerdown', event => {
    if (!event.target.matches('#comparison-n img, #comparison-s img')) return;
    const style = getComputedStyle(event.target);
    pan = {x: parseFloat(style.getPropertyValue('--pan-x')) || 0, y: parseFloat(style.getPropertyValue('--pan-y')) || 0};
    drag = {x: event.clientX - pan.x, y: event.clientY - pan.y};
    event.preventDefault();
  });
  document.addEventListener('pointermove', event => {
    if (!drag) return;
    pan = {x: event.clientX - drag.x, y: event.clientY - drag.y};
    images().forEach(img => {
      img.style.setProperty('--pan-x', `${pan.x}px`);
      img.style.setProperty('--pan-y', `${pan.y}px`);
    });
  });
  document.addEventListener('pointerup', () => { drag = null; });
  document.addEventListener('pointercancel', () => { drag = null; });
  window.addEventListener('blur', () => { drag = null; });
  document.addEventListener('dblclick', event => {
    if (!event.target.matches('#comparison-n img, #comparison-s img')) return;
    pan = {x:0,y:0};
    images().forEach(img => { img.style.setProperty('--pan-x','0px'); img.style.setProperty('--pan-y','0px'); });
  });
}
