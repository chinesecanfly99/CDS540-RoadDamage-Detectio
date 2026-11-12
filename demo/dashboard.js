async () => {
  const {changeLocale} = await import(new URL('__GRADIO_I18N_MODULE__', document.baseURI).href);
  window.roadReviewLocale = language => {
    changeLocale(language === 'zh' ? 'zh-CN' : 'en');
    document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
    document.title = language === 'zh' ? '道路损伤检测与人工复核' : 'Road Damage Detection & Review';
  };
  // The app.load handler sets English after all lazy components have mounted.
  // Event ids and session state change when the local service restarts. Reload
  // an old page instead of leaving it attached to obsolete queue handlers.
  let serviceId;
  let checkingService = false;
  async function checkService() {
    if (checkingService) return;
    checkingService = true;
    try {
      const response = await fetch(new URL('config', document.baseURI), {cache:'no-store'});
      if (!response.ok) return;
      const config = await response.json();
      const id = config.app_id;
      if (id !== undefined) {
        if (serviceId !== undefined && serviceId !== id) window.location.reload();
        serviceId = id;
      }
    } catch (_) { /* Wait for a temporarily restarting local service. */ }
    finally { checkingService = false; }
  }
  await checkService();
  const serviceTimer = setInterval(checkService, 10000);
  window.addEventListener('beforeunload', () => clearInterval(serviceTimer), {once:true});
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
