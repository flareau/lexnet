const columns = {
  word: [['entry_name','Entry'], ['std_name','Lexical unit'], ['pos','POS']],
  lf: [['lf_name','Function'], ['source_name','Source'], ['target_name','Target'], ['form','Form']],
  feature: [['entry_name','Entry'], ['std_name','Lexical unit'], ['matching_features','Matching features']]
};
let kind = 'word';
let currentRows = [];
let sortKey = null;
let sortAscending = true;
const collator = new Intl.Collator(undefined, {numeric: true, sensitivity: 'base'});
const status = document.querySelector('#status');
const results = document.querySelector('#results');
const head = document.querySelector('#head');

function selectTab(next) {
  kind = next;
  currentRows = []; sortKey = null; sortAscending = true;
  document.querySelectorAll('.tab').forEach(x => x.classList.toggle('active', x.dataset.kind === kind));
  document.querySelectorAll('.panel').forEach(x => x.classList.toggle('active', x.dataset.kind === kind));
  renderHead(); results.replaceChildren(); status.textContent = 'Ready.';
  document.querySelector(`#${kind}-panel input[name=q]`).focus();
}
function renderHead() {
  head.replaceChildren(...columns[kind].map(([key, label]) => {
    const th=document.createElement('th');
    th.tabIndex=0;
    th.textContent=label + (sortKey === key ? (sortAscending ? ' ▲' : ' ▼') : '');
    th.setAttribute('aria-sort', sortKey === key ? (sortAscending ? 'ascending' : 'descending') : 'none');
    th.addEventListener('click', () => sortBy(key));
    th.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); sortBy(key); } });
    return th;
  }));
}
function sortBy(key) {
  if (sortKey === key) sortAscending = !sortAscending;
  else { sortKey = key; sortAscending = true; }
  currentRows.sort((left, right) => {
    const a = String(left[key] || ''); const b = String(right[key] || '');
    if (!a && b) return 1; if (a && !b) return -1;
    return collator.compare(a, b) * (sortAscending ? 1 : -1);
  });
  renderHead(); renderRows();
}
function renderRows() {
  const fragment = document.createDocumentFragment();
  for (const row of currentRows) {
    const tr = document.createElement('tr');
    for (const [key] of columns[kind]) {
      const td=document.createElement('td'); td.title=row[key] || '';
      if ((kind === 'word' || kind === 'feature') && (key === 'entry_name' || key === 'std_name')) {
        const link=document.createElement('button');
        link.type='button'; link.className='node-link'; link.textContent=row[key] || '';
        const itemType = key === 'entry_name' ? 'entry' : 'node';
        const itemId = key === 'entry_name' ? row.entry_id : row.node_id;
        link.addEventListener('click', () => showItem(itemType, itemId));
        td.append(link);
      } else if (kind === 'lf' && (key === 'source_name' || key === 'target_name' || (key === 'form' && row.form))) {
        const link=document.createElement('button');
        link.type='button'; link.className='node-link'; link.textContent=row[key] || '';
        const nodeId = key === 'source_name' ? row.source_node_id : row.target_node_id;
        link.addEventListener('click', event => { event.stopPropagation(); showItem('node', nodeId); });
        td.append(link);
      } else td.textContent=row[key] || '';
      tr.append(td);
    }
    fragment.append(tr);
  }
  results.replaceChildren(fragment);
}
async function search(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const params = new URLSearchParams(new FormData(form));
  params.set('kind', form.dataset.kind);
  if (form.dataset.kind === 'word') params.set('forms', form.elements.forms.checked ? '1' : '0');
  status.textContent = 'Searching…'; results.replaceChildren();
  try {
    const response = await fetch('/api/search?' + params);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Search failed');
    status.textContent = payload.status;
    currentRows = payload.rows; sortKey = null; sortAscending = true;
    renderHead(); renderRows();
  } catch (error) { status.textContent = error.message; }
}
async function showItem(itemType, itemId) {
  const params = new URLSearchParams({type: itemType, id: itemId});
  const response = await fetch('/api/inspect?' + params);
  const payload = await response.json();
  renderDetails(payload);
}
function renderDetails(payload) {
  const details = document.querySelector('#details');
  document.querySelector('#inspector-title').textContent = payload.title || 'Inspector';
  if (!payload.description) { details.textContent = payload.error || 'Unable to load item.'; return; }
  const links = new Map();
  for (const link of payload.links || []) {
    if (!links.has(link.line)) links.set(link.line, []);
    links.get(link.line).push(link);
  }
  const content = document.createDocumentFragment();
  const lines = payload.description.split('\n');
  lines.forEach((line, index) => {
    const choices = links.get(line);
    const relation = choices && choices.shift();
    if (relation) {
      content.append(document.createTextNode(relation.prefix));
      const link = document.createElement('button');
      link.type='button'; link.className='node-link'; link.textContent=relation.item_name;
      link.addEventListener('click', () => showItem(relation.item_type, relation.item_id));
      content.append(link);
      content.append(document.createTextNode(relation.suffix || ''));
    } else content.append(document.createTextNode(line));
    if (index < lines.length - 1) content.append(document.createTextNode('\n'));
  });
  details.replaceChildren(content);
}
async function initialize() {
  const response = await fetch('/api/meta'); const meta = await response.json();
  document.querySelector('#dataset').textContent = `${meta.path} — ${meta.entries.toLocaleString()} entries, ${meta.units.toLocaleString()} units`;
  for (const [target, values] of [['lf-names', meta.lexical_functions], ['feature-names', meta.features]]) {
    const list=document.querySelector('#'+target); for (const value of values) { const option=document.createElement('option'); option.value=value; list.append(option); }
  }
  const familyControl = document.querySelector('#lf-family');
  const familyToggle = familyControl.querySelector('.tree-select-toggle');
  const familyMenu = familyControl.querySelector('.tree-select-menu');
  const familyValue = familyControl.querySelector('input[name=family_id]');
  const functionSelect = document.querySelector('#lf-function');
  const nameInput = document.querySelector('#lf-panel input[name=q]');
  function chooseFamily(value, label, clearName=true) {
    familyValue.value=value; familyToggle.textContent=label;
    familyMenu.hidden=true; familyToggle.setAttribute('aria-expanded', 'false');
    if (clearName) nameInput.value='';
    if (value.startsWith('group:')) {
      const groupNumber=value.slice(6);
      functionSelect.replaceChildren();
      const all=document.createElement('option'); all.value=''; all.textContent=`All functions in Group ${groupNumber}`; functionSelect.append(all);
      functionSelect.disabled=true;
      return;
    }
    const family = meta.lf_hierarchy.flatMap(group => group.families).find(item => item.id === value);
    functionSelect.replaceChildren();
    const all=document.createElement('option'); all.value=''; all.textContent=family ? 'All functions in family' : 'Choose a family first'; functionSelect.append(all);
    for (const lf of family?.functions || []) { const option=document.createElement('option'); option.value=lf.id; option.textContent=lf.name; functionSelect.append(option); }
    functionSelect.disabled = !family;
  }
  for (const group of meta.lf_hierarchy) {
    const groupItem=document.createElement('button'); groupItem.type='button'; groupItem.className='tree-select-item tree-select-group'; groupItem.setAttribute('role', 'option'); groupItem.textContent=`Group ${group.index}`;
    groupItem.addEventListener('click', () => chooseFamily(`group:${group.index}`, `Group ${group.index}`)); familyMenu.append(groupItem);
    for (const family of group.families) {
      const familyItem=document.createElement('button'); familyItem.type='button'; familyItem.className='tree-select-item tree-select-family'; familyItem.setAttribute('role', 'option'); familyItem.textContent=family.name;
      familyItem.addEventListener('click', () => chooseFamily(family.id, family.name)); familyMenu.append(familyItem);
    }
  }
  familyToggle.addEventListener('click', () => {
    familyMenu.hidden=!familyMenu.hidden;
    familyToggle.setAttribute('aria-expanded', String(!familyMenu.hidden));
  });
  document.addEventListener('click', event => { if (!familyControl.contains(event.target)) { familyMenu.hidden=true; familyToggle.setAttribute('aria-expanded', 'false'); } });
  functionSelect.addEventListener('change', () => { nameInput.value = ''; });
  nameInput.addEventListener('input', () => {
    if (!nameInput.value) return;
    chooseFamily('', 'Choose a family…', false);
  });
}
document.querySelectorAll('.tab').forEach(x => x.addEventListener('click', () => selectTab(x.dataset.kind)));
document.querySelectorAll('form').forEach(x => x.addEventListener('submit', search));
renderHead(); initialize();
