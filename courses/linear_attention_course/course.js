/* Only tab navigation. All teaching content, diagrams and answers are static. */
(() => {
  const panels = [...document.querySelectorAll('[role="tabpanel"]')];
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  if (!panels.length) return;
  function select(id, focus = false) {
    const active = panels.find(p => p.id === id) || panels[0];
    panels.forEach(p => { p.hidden = p !== active; });
    tabs.forEach(t => {
      const on = t.getAttribute('aria-controls') === active.id;
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
      if (on && focus) t.focus();
    });
  }
  function navigate(id, focus = false) {
    history.replaceState(null, '', '#' + id);
    select(id, focus);
  }
  tabs.forEach((tab, i) => {
    tab.addEventListener('click', () => navigate(tab.getAttribute('aria-controls')));
    tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (i + 1) % tabs.length;
      if (event.key === 'ArrowLeft') next = (i + tabs.length - 1) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      if (next !== undefined) {
        event.preventDefault();
        navigate(tabs[next].getAttribute('aria-controls'), true);
      }
    });
  });
  document.querySelectorAll('[data-next]').forEach(button => {
    button.addEventListener('click', () => {
      navigate(button.dataset.next, true);
      document.querySelector('.nav').scrollIntoView({block: 'start'});
    });
  });
  window.addEventListener('hashchange', () => select(location.hash.slice(1)));
  select(location.hash.slice(1));
})();
