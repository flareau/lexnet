const columns = {
  word: [['entry_name','Entry'], ['std_name','Lexical unit'], ['pos','POS']],
  lf: [['lf_name','Function'], ['source_name','Source'], ['target_name','Target'], ['form','Form']],
  feature: [['entry_name','Entry'], ['std_name','Lexical unit'], ['matching_features','Matching features']],
  semantic: [['semantic_label_name','Label'], ['std_name','Lexical unit']]
};
let kind = 'word';
let currentRows = [];
let sortKey = null;
let sortAscending = true;
const collator = new Intl.Collator(undefined, {numeric: true, sensitivity: 'base'});
const status = document.querySelector('#status');
const results = document.querySelector('#results');
const head = document.querySelector('#head');

const atomicLfNames = ['De_nouveau'];
let copolysemyGraphCounter = 0;

function appendLfName(parent, name, semanticSegments=null) {
  const container = document.createElement('span');
  container.className = 'lf-name';
  if (Array.isArray(semanticSegments) && semanticSegments.some(segment => segment.actant)) {
    for (const segment of semanticSegments) {
      if (segment.actant) {
        const variable = document.createElement('span');
        variable.className = 'semantic-variable'; variable.textContent = segment.text;
        variable.title = segment.actant;
        variable.setAttribute('aria-label', `${segment.actant}: ${segment.text}`);
        container.append(variable);
      } else container.append(document.createTextNode(segment.text));
    }
    parent.append(container);
    return;
  }
  let plainText = '';
  const flushText = () => {
    if (!plainText) return;
    container.append(document.createTextNode(plainText));
    plainText = '';
  };
  for (let index = 0; index < name.length;) {
    const atom = atomicLfNames.find(candidate => name.startsWith(candidate, index));
    if (atom) {
      plainText += atom;
      index += atom.length;
      continue;
    }
    const marker = name[index];
    if (marker !== '_' && marker !== '^') {
      plainText += marker;
      index += 1;
      continue;
    }
    flushText();
    const scripts = document.createElement('span');
    scripts.className = 'lf-name-scripts';
    while (name[index] === '_' || name[index] === '^') {
      const scriptMarker = name[index];
      index += 1;
      let end = index;
      if (scriptMarker === '^') {
        const roman = name.slice(index).match(/^(?:I\/II|III|II|I)(?![a-z])/);
        if (roman) end += roman[0].length;
      }
      if (end === index) {
        while (end < name.length) {
          const character = name[end];
          if (character === '_' || character === '^' || character === '•' || /[A-Z]/.test(character)) break;
          end += 1;
        }
      }
      const script = document.createElement(scriptMarker === '_' ? 'sub' : 'sup');
      script.textContent = name.slice(index, end).trimEnd();
      scripts.append(script);
      index = end;
    }
    container.append(scripts);
  }
  flushText();
  parent.append(container);
}

function appendLexicalName(parent, label) {
  const container = document.createElement('span');
  container.className = 'lexical-name';
  if (label.confidence < 100) {
    container.classList.add('low-confidence');
    container.title = `Confidence: ${label.confidence}%`;
  }
  const base = document.createElement('span');
  base.className = 'lexical-name-base'; base.textContent = label.name || '';
  container.append(base);
  const scripts = document.createElement('span');
  scripts.className = 'lexical-name-scripts';
  if (label.superscript) {
    const superscript = document.createElement('sup');
    superscript.className = 'lexical-name-homograph'; superscript.textContent = label.superscript;
    scripts.append(superscript);
  }
  if (label.subscript || label.lexnum) {
    const subscripts = document.createElement('sub');
    subscripts.className = 'lexical-name-subscripts';
    if (label.subscript) {
      const grammar = document.createElement('span');
      grammar.className = 'lexical-name-grammar'; grammar.textContent = label.subscript;
      subscripts.append(grammar);
    }
    if (label.lexnum) {
      const sense = document.createElement('span');
      sense.className = 'lexical-name-sense'; sense.textContent = label.lexnum;
      subscripts.append(sense);
    }
    scripts.append(subscripts);
  }
  if (scripts.childNodes.length) container.append(scripts);
  parent.append(container);
}

function appendSemanticSegments(parent, segments) {
  for (const segment of segments || []) {
    if (segment.actant) {
      const variable = document.createElement('span');
      variable.className = 'semantic-variable';
      variable.textContent = segment.text;
      variable.title = segment.actant;
      variable.setAttribute('aria-label', `${segment.actant}: ${segment.text}`);
      parent.append(variable);
    } else parent.append(document.createTextNode(segment.text));
  }
}

function selectTab(next) {
  kind = next;
  currentRows = []; sortKey = null; sortAscending = true;
  document.querySelector('.left').classList.toggle('semantic-mode', kind === 'semantic');
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
      if ((kind === 'word' || kind === 'feature' || kind === 'semantic') && (key === 'entry_name' || key === 'std_name')) {
        const link=document.createElement('button');
        link.type='button'; link.className='node-link';
        appendLexicalName(link, key === 'entry_name' ? row.entry_label : row.unit_label);
        const itemType = key === 'entry_name' ? 'entry' : 'node';
        const itemId = key === 'entry_name' ? row.entry_id : row.node_id;
        link.addEventListener('click', () => showItem(itemType, itemId));
        td.append(link);
      } else if (kind === 'lf' && (key === 'source_name' || key === 'target_name')) {
        const link=document.createElement('button');
        link.type='button'; link.className='node-link';
        appendLexicalName(link, key === 'source_name' ? row.source_label : row.target_label);
        const nodeId = key === 'source_name' ? row.source_node_id : row.target_node_id;
        link.addEventListener('click', event => { event.stopPropagation(); showItem('node', nodeId); });
        td.append(link);
      } else if (kind === 'lf' && key === 'form' && row.form) {
        const link=document.createElement('button');
        link.type='button'; link.className='node-link'; link.textContent=row.form;
        link.addEventListener('click', event => { event.stopPropagation(); showItem('node', row.target_node_id); });
        td.append(link);
      } else if (kind === 'lf' && key === 'lf_name') {
        appendLfName(td, row[key] || '', row.lf_name_segments);
      } else if (kind === 'semantic' && key === 'semantic_label_name') {
        const link=document.createElement('button');
        link.type='button'; link.className='node-link'; link.textContent=row.semantic_label_name;
        if (Number(row.confidence) < 100) {
          link.classList.add('low-confidence'); link.title=`Confidence: ${row.confidence}%`;
        }
        link.addEventListener('click', () => showItem('semantic_label', row.semantic_label_id));
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
  if (form.dataset.kind === 'semantic') params.set('descendants', form.elements.descendants.checked ? '1' : '0');
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
  const inspector = document.querySelector('#details');
  document.querySelector('#inspector-title').textContent = payload.title || 'Inspector';
  if (!payload.description) { inspector.textContent = payload.error || 'Unable to load item.'; return; }
  const links = new Map();
  for (const link of payload.links || []) {
    if (!links.has(link.line)) links.set(link.line, []);
    links.get(link.line).push(link);
  }
  const content = document.createDocumentFragment();
  const lines = payload.description.split('\n');
  const sectionTitles = new Set([
    'ENTRY INFORMATION', 'LEXICAL UNITS', 'GRAMMATICAL INFORMATION',
    'DEFINITION', 'WORDFORMS', 'SEMANTIC LABELS', 'PROPOSITIONAL FORMS',
    'LEXICAL RELATIONS', 'COPOLYSEMY GRAPH', 'EXAMPLES', 'LABEL INFORMATION', 'CLASSIFICATION',
    'CLASS INFORMATION', 'PARENTS', 'SUBCLASSES', 'DIRECT LABELS'
  ]);
  function appendLine(parent, line, removeBullet=false) {
    const choices = links.get(line);
    const relation = choices && choices.shift();
    if (relation) {
      if (relation.function_name) {
        appendLfName(parent, relation.function_name, relation.function_segments);
        parent.append(document.createTextNode(': '));
      } else {
        const prefix = removeBullet && relation.prefix.startsWith(' • ')
          ? relation.prefix.slice(3) : relation.prefix;
        parent.append(document.createTextNode(prefix));
      }
      if (relation.example_segments) {
        const example = document.createElement('span');
        example.className = 'example-text';
        if (relation.low_confidence) {
          example.classList.add('low-confidence');
          example.title = `Confidence: ${relation.confidence}%`;
        }
        for (const segment of relation.example_segments) {
          if (segment.highlighted) {
            const mark = document.createElement('mark');
            mark.className = 'example-occurrence'; mark.textContent = segment.text;
            example.append(mark);
          } else example.append(document.createTextNode(segment.text));
        }
        parent.append(example);
      } else if (relation.plain_item_label !== undefined) {
        const link = document.createElement('button');
        link.type = 'button'; link.className = 'node-link';
        link.textContent = relation.plain_item_label;
        if (relation.low_confidence) {
          link.classList.add('low-confidence');
          link.title = `Confidence: ${relation.confidence}%`;
        }
        link.addEventListener('click', () => showItem(relation.item_type, relation.item_id));
        parent.append(link);
      } else if (relation.information_text !== undefined) {
        const information = document.createElement('span');
        if (relation.information_segments) {
          appendSemanticSegments(information, relation.information_segments);
        } else information.textContent = relation.information_text;
        if (relation.low_confidence) {
          information.className = 'low-confidence';
          information.title = `Confidence: ${relation.confidence}%`;
        }
        parent.append(information);
      } else if (relation.items) {
        const list = document.createElement('ul'); list.className = 'lf-values';
        for (const item of relation.items) {
          const listItem = document.createElement('li');
          if (item.merged) {
            const merged = document.createElement('span');
            merged.className = 'lf-merge'; merged.title = 'Merged value';
            merged.setAttribute('aria-label', 'Merged value');
            listItem.append(merged);
          }
          const link = document.createElement('button');
          link.type='button'; link.className='node-link';
          appendLexicalName(link, item.item_label);
          link.addEventListener('click', () => showItem(item.item_type, item.item_id));
          listItem.append(link);
          if (item.frame) {
            const frame = document.createElement('span');
            frame.className = 'lf-frame'; frame.textContent = item.frame;
            frame.title = 'Syntactic frame'; frame.setAttribute('aria-label', `Syntactic frame: ${item.frame}`);
            listItem.append(frame);
          }
          if (item.constraint) {
            const constraint = document.createElement('span');
            constraint.className = 'lf-constraint'; constraint.textContent = item.constraint;
            constraint.title = 'Constraint'; constraint.setAttribute('aria-label', `Constraint: ${item.constraint}`);
            listItem.append(constraint);
          }
          list.append(listItem);
        }
        parent.append(list);
      } else {
        if (relation.item_type) {
          const link = document.createElement('button');
          link.type='button'; link.className='node-link';
          appendLexicalName(link, relation.item_label);
          link.addEventListener('click', () => showItem(relation.item_type, relation.item_id));
          parent.append(link);
          if (relation.unit_annotations) {
            parent.append(document.createTextNode(' ('));
            relation.unit_annotations.labels.forEach((label, index) => {
              if (index) parent.append(document.createTextNode(', '));
              const labelLink = document.createElement('button');
              labelLink.type = 'button'; labelLink.className = 'node-link';
              labelLink.textContent = label.text;
              if (label.low_confidence) {
                labelLink.classList.add('low-confidence');
                labelLink.title = `Confidence: ${label.confidence}%`;
              }
              labelLink.addEventListener('click', () => showItem('semantic_label', label.item_id));
              parent.append(labelLink);
            });
            if (relation.unit_annotations.labels.length && relation.unit_annotations.propforms.length) {
              parent.append(document.createTextNode(' : '));
            }
            relation.unit_annotations.propforms.forEach((propform, index) => {
              if (index) parent.append(document.createTextNode(', '));
              const value = document.createElement('span');
              if (propform.low_confidence) {
                value.className = 'low-confidence';
                value.title = `Confidence: ${propform.confidence}%`;
              }
              appendSemanticSegments(value, propform.segments);
              parent.append(value);
            });
            parent.append(document.createTextNode(')'));
          }
        } else appendLexicalName(parent, relation.label);
      }
      if (!relation.unit_annotations) parent.append(document.createTextNode(relation.suffix || ''));
    } else parent.append(document.createTextNode(removeBullet ? line.slice(3) : line));
  }
  function appendRelationGroups(parent, relationLines) {
    let list = null;
    for (const line of relationLines) {
      if (line === 'Outgoing' || line === 'Incoming') {
        const group = document.createElement('section'); group.className = 'relation-group';
        const heading = document.createElement('h3'); heading.textContent = line;
        list = document.createElement('ul'); list.className = 'inspector-list';
        group.append(heading, list); parent.append(group);
      } else if (list) {
        const item = document.createElement('li'); appendLine(item, line); list.append(item);
      }
    }
  }
  function createCopolysemyNode(itemId, itemLabel) {
    const link = document.createElement('button');
    link.type = 'button'; link.className = 'node-link copolysemy-node';
    appendLexicalName(link, itemLabel);
    link.addEventListener('click', () => showItem('node', itemId));
    return link;
  }
  function appendCopolysemyGraph(parent, graph) {
    const nodes = graph?.nodes || [];
    const edges = graph?.edges || [];
    if (!nodes.length) {
      const empty = document.createElement('p');
      empty.className = 'copolysemy-empty'; empty.textContent = 'No lexical units.';
      parent.append(empty);
      return;
    }
    if (!window.dagre) {
      const error = document.createElement('p');
      error.className = 'copolysemy-empty'; error.textContent = 'Graph layout is unavailable.';
      parent.append(error);
      return;
    }

    const measurement = document.createElement('div');
    measurement.className = 'copolysemy-measurement'; document.body.append(measurement);
    const dimensions = new Map();
    for (const node of nodes) {
      const link = createCopolysemyNode(node.item_id, node.item_label);
      measurement.replaceChildren(link);
      const bounds = link.getBoundingClientRect();
      dimensions.set(node.item_id, {
        width: Math.min(260, Math.max(68, Math.ceil(bounds.width) + 6)),
        height: Math.max(30, Math.ceil(bounds.height) + 8),
      });
    }
    measurement.remove();

    const layout = new window.dagre.graphlib.Graph({multigraph: true})
      .setGraph({
        rankdir: 'TB', acyclicer: 'greedy', ranker: 'network-simplex',
        nodesep: 30, edgesep: 18, ranksep: 66, marginx: 18, marginy: 18,
      })
      .setDefaultEdgeLabel(() => ({}));
    for (const node of nodes) {
      layout.setNode(node.item_id, {...dimensions.get(node.item_id), node});
    }
    edges.forEach((edge, index) => {
      const label = edge.type_name + (edge.subtype_name ? ` · ${edge.subtype_name}` : '');
      layout.setEdge(edge.source_id, edge.target_id, {
        edge, label, width: Math.min(190, Math.max(54, label.length * 6.5 + 10)), height: 18,
        labelpos: 'c', labeloffset: 0,
      }, `edge-${index}`);
    });
    window.dagre.layout(layout);

    const namespace = 'http://www.w3.org/2000/svg';
    const svgElement = (name, attributes={}) => {
      const element = document.createElementNS(namespace, name);
      for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
      return element;
    };
    const graphId = `copolysemy-${++copolysemyGraphCounter}`;
    const bounds = layout.graph();
    const viewport = document.createElement('div'); viewport.className = 'copolysemy-graph';
    const svg = svgElement('svg', {
      class: 'copolysemy-svg', width: Math.ceil(bounds.width), height: Math.ceil(bounds.height),
      viewBox: `0 0 ${Math.ceil(bounds.width)} ${Math.ceil(bounds.height)}`,
      role: 'img', 'aria-label': 'Copolysemy graph',
    });
    const definitions = svgElement('defs');
    const edgeColor = semantics => ['#aeb9c4', '#6d7882', '#17324d'][semantics] || '#536475';
    for (const semantics of ['default', 0, 1, 2]) {
      const marker = svgElement('marker', {
        id: `${graphId}-arrow-${semantics}`, viewBox: '0 0 8 8', refX: 7, refY: 4,
        markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse',
      });
      marker.append(svgElement('path', {
        d: 'M 0 0 L 8 4 L 0 8 z', fill: semantics === 'default' ? '#536475' : edgeColor(semantics),
      }));
      definitions.append(marker);
    }
    svg.append(definitions);

    for (const edgeKey of layout.edges()) {
      const positioned = layout.edge(edgeKey);
      const edge = positioned.edge;
      const semantics = [0, 1, 2].includes(edge.semantics) ? edge.semantics : 'default';
      const path = svgElement('path', {
        class: 'copolysemy-svg-edge' + (edge.derivation === false ? ' copolysemy-nonderivational' : ''),
        d: positioned.points.map((point, index) => `${index ? 'L' : 'M'} ${point.x} ${point.y}`).join(' '),
        stroke: semantics === 'default' ? '#536475' : edgeColor(semantics),
        'marker-end': `url(#${graphId}-arrow-${semantics})`,
      });
      const metadata = [];
      if (edge.semantics !== null) metadata.push(`Semantic weight: ${edge.semantics}`);
      if (edge.derivation !== null) metadata.push(edge.derivation ? 'Derivational' : 'Non-derivational');
      if (metadata.length) {
        const title = svgElement('title'); title.textContent = metadata.join('; '); path.append(title);
      }
      svg.append(path);
      const label = svgElement('g', {class: 'copolysemy-svg-edge-label'});
      const text = svgElement('text', {x: positioned.x, y: positioned.y, dy: '.35em'});
      text.textContent = positioned.label; label.append(text); svg.append(label);
    }

    for (const nodeId of layout.nodes()) {
      const positioned = layout.node(nodeId);
      const foreignObject = svgElement('foreignObject', {
        x: positioned.x - positioned.width / 2, y: positioned.y - positioned.height / 2,
        width: positioned.width, height: positioned.height,
      });
      const wrapper = document.createElement('div'); wrapper.className = 'copolysemy-svg-node';
      wrapper.append(createCopolysemyNode(nodeId, positioned.node.item_label));
      foreignObject.append(wrapper); svg.append(foreignObject);
    }
    viewport.append(svg); parent.append(viewport);
    if (edges.length) {
      const legend = document.createElement('aside');
      legend.className = 'copolysemy-legend'; legend.setAttribute('aria-label', 'Graph legend');
      const addGroup = (title, items) => {
        const group = document.createElement('div');
        const heading = document.createElement('span'); heading.textContent = title;
        const list = document.createElement('ul');
        for (const [className, text] of items) {
          const item = document.createElement('li');
          const swatch = document.createElement('span');
          swatch.className = `copolysemy-legend-line ${className}`;
          swatch.setAttribute('aria-hidden', 'true');
          item.append(swatch, document.createTextNode(text)); list.append(item);
        }
        group.append(heading, list); legend.append(group);
      };
      addGroup('Semantic weight', [
        ['copolysemy-legend-weight-0', '0 · none'],
        ['copolysemy-legend-weight-1', '1 · intermediate'],
        ['copolysemy-legend-weight-2', '2 · maximum'],
      ]);
      addGroup('Derivation', [
        ['', 'derivational'],
        ['copolysemy-legend-nonderivational', 'non-derivational'],
      ]);
      parent.append(legend);
    }
  }
  let index = 0;
  while (index < lines.length) {
    if (sectionTitles.has(lines[index])) {
      const title = lines[index];
      let end = index + 1;
      while (end < lines.length && !sectionTitles.has(lines[end])) end += 1;
      const sectionLines = lines.slice(index + 1, end);
      while (sectionLines.at(-1) === '') sectionLines.pop();
      const section = document.createElement('details');
      section.className = 'inspector-section';
      section.open = title !== 'WORDFORMS' && title !== 'ENTRY INFORMATION';
      const summary = document.createElement('summary');
      const itemCount = title === 'COPOLYSEMY GRAPH'
        ? (payload.copolysemy_graph?.edges || []).length
        : sectionLines.filter(line => line && line !== 'Outgoing' && line !== 'Incoming').length;
      summary.textContent = `${title} (${itemCount})`;
      const body = document.createElement('div');
      body.className = 'inspector-section-content';
      const manualBullets = sectionLines.length > 0 && sectionLines.every(line => line.startsWith(' • '));
      if (title === 'LEXICAL RELATIONS') {
        appendRelationGroups(body, sectionLines);
      } else if (title === 'COPOLYSEMY GRAPH') {
        appendCopolysemyGraph(body, payload.copolysemy_graph);
      } else if (manualBullets) {
        const list = document.createElement('ul'); list.className = 'inspector-list';
        for (const line of sectionLines) {
          const item = document.createElement('li'); appendLine(item, line, manualBullets); list.append(item);
        }
        body.append(list);
      } else {
        sectionLines.forEach((line, lineIndex) => {
          appendLine(body, line);
          if (lineIndex < sectionLines.length - 1) body.append(document.createTextNode('\n'));
        });
      }
      section.append(summary, body); content.append(section);
      if (end < lines.length) content.append(document.createTextNode('\n'));
      index = end;
      continue;
    }
    appendLine(content, lines[index]);
    if (index < lines.length - 1) content.append(document.createTextNode('\n'));
    index += 1;
  }
  inspector.replaceChildren(content);
}
async function initialize() {
  const response = await fetch('/api/meta'); const meta = await response.json();
  document.querySelector('#dataset').textContent = `${meta.path} — ${meta.entries.toLocaleString()} entries, ${meta.units.toLocaleString()} units`;
  for (const [target, values] of [['lf-names', meta.lexical_functions], ['feature-names', meta.features]]) {
    const list=document.querySelector('#'+target); for (const value of values) { const option=document.createElement('option'); option.value=value; list.append(option); }
  }
  const semanticNames = new Set(meta.semantic_labels || []);
  for (const item of meta.semantic_hierarchy || []) semanticNames.add(item.name);
  const semanticNameList = document.querySelector('#semantic-label-names');
  for (const value of [...semanticNames].sort(collator.compare)) {
    const option=document.createElement('option'); option.value=value; semanticNameList.append(option);
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

  const semanticForm = document.querySelector('#semantic-panel');
  const semanticInput = semanticForm.querySelector('input[name=q]');
  const semanticClassValue = semanticForm.querySelector('input[name=class_id]');
  const semanticLabelValue = semanticForm.querySelector('input[name=semantic_label_id]');
  const semanticResults = document.querySelector('.results-area');
  const semanticBrowser = document.querySelector('#semantic-browser');
  const semanticResizer = document.querySelector('#semantic-resizer');
  const semanticTree = document.querySelector('#semantic-tree');
  const rootList = document.createElement('ul');
  semanticTree.append(rootList);
  const levelLists = [rootList];

  function selectSemantic(itemType, itemId) {
    semanticInput.value = '';
    semanticClassValue.value = itemType === 'semantic_class' ? itemId : '';
    semanticLabelValue.value = itemType === 'semantic_label' ? itemId : '';
    semanticTree.querySelectorAll('.selected').forEach(item => item.classList.remove('selected'));
    semanticTree.querySelectorAll(`[data-item-id="${CSS.escape(itemId)}"]`).forEach(item => item.classList.add('selected'));
    filterSemanticTree('');
    semanticForm.requestSubmit();
    showItem(itemType, itemId);
  }

  for (const item of meta.semantic_hierarchy || []) {
    levelLists.length = item.depth + 1;
    const parentList = levelLists[item.depth] || rootList;
    const listItem = document.createElement('li');
    const details = document.createElement('details');
    details.open = item.depth < 2;
    const summary = document.createElement('summary');
    const button = document.createElement('button');
    button.type = 'button'; button.className = 'semantic-tree-node';
    if (item.semantic_field) button.classList.add('semantic-field');
    button.dataset.itemId = item.id; button.textContent = item.name;
    button.addEventListener('click', event => {
      event.preventDefault(); event.stopPropagation(); selectSemantic('semantic_class', item.id);
    });
    summary.append(button);
    if (item.multiple_paths) {
      const multiple = document.createElement('span');
      multiple.className = 'semantic-multiple'; multiple.textContent = '⧉';
      multiple.title = 'Class with multiple parents'; summary.append(multiple);
    }
    const count = document.createElement('span');
    count.className = 'semantic-tree-count'; count.textContent = item.count.toLocaleString();
    summary.append(count);
    const children = document.createElement('ul');
    for (const label of item.labels || []) {
      const labelItem = document.createElement('li');
      labelItem.className = 'semantic-label-row';
      const labelButton = document.createElement('button');
      labelButton.type = 'button'; labelButton.className = 'semantic-tree-node semantic-tree-label';
      labelButton.dataset.itemId = label.id; labelButton.textContent = label.name;
      labelButton.addEventListener('click', () => selectSemantic('semantic_label', label.id));
      const labelCount = document.createElement('span');
      labelCount.className = 'semantic-tree-count'; labelCount.textContent = label.count.toLocaleString();
      labelItem.append(labelButton, labelCount); children.append(labelItem);
    }
    details.append(summary, children); listItem.append(details); parentList.append(listItem);
    levelLists[item.depth + 1] = children;
  }

  function filterSemanticTree(value) {
    const needle = value.trim().toLocaleLowerCase();
    const items = [...semanticTree.querySelectorAll('li')];
    items.forEach(item => { item.hidden = Boolean(needle); });
    if (!needle) return;
    for (const button of semanticTree.querySelectorAll('.semantic-tree-node')) {
      if (!button.textContent.toLocaleLowerCase().includes(needle)) continue;
      let item = button.closest('li');
      while (item) {
        item.hidden = false;
        const parentDetails = item.parentElement.closest('details');
        if (parentDetails) parentDetails.open = true;
        item = item.parentElement.closest('li');
      }
    }
  }

  semanticInput.addEventListener('input', () => {
    semanticClassValue.value = ''; semanticLabelValue.value = '';
    semanticTree.querySelectorAll('.selected').forEach(item => item.classList.remove('selected'));
    filterSemanticTree(semanticInput.value);
  });
  semanticForm.elements.descendants.addEventListener('change', () => {
    if (semanticClassValue.value) semanticForm.requestSubmit();
  });
  document.querySelector('#semantic-expand').addEventListener('click', () => {
    semanticTree.querySelectorAll('details').forEach(item => { item.open = true; });
  });
  document.querySelector('#semantic-collapse').addEventListener('click', () => {
    semanticTree.querySelectorAll('details').forEach(item => { item.open = false; });
  });

  function setSemanticBrowserWidth(width, persist=false) {
    const available = semanticResults.getBoundingClientRect().width;
    const maximum = Math.max(240, available - 227);
    const next = Math.min(Math.max(240, width), maximum);
    semanticResults.style.setProperty('--semantic-tree-width', `${next}px`);
    semanticResizer.setAttribute('aria-valuenow', String(Math.round(next)));
    semanticResizer.setAttribute('aria-valuemax', String(Math.round(maximum)));
    if (persist) {
      try { localStorage.setItem('lexnet-semantic-tree-width', String(Math.round(next))); }
      catch (error) { /* Storage may be disabled; resizing still works. */ }
    }
  }

  let savedSemanticWidth = 0;
  try { savedSemanticWidth = Number(localStorage.getItem('lexnet-semantic-tree-width')); }
  catch (error) { /* Use the CSS default. */ }
  requestAnimationFrame(() => setSemanticBrowserWidth(
    savedSemanticWidth > 0 ? savedSemanticWidth : semanticResults.getBoundingClientRect().width * .55,
  ));

  let resizeStartX = 0;
  let resizeStartWidth = 0;
  semanticResizer.addEventListener('pointerdown', event => {
    resizeStartX = event.clientX;
    resizeStartWidth = semanticBrowser.getBoundingClientRect().width;
    semanticResizer.classList.add('dragging');
    semanticResizer.setPointerCapture(event.pointerId);
    event.preventDefault();
  });
  semanticResizer.addEventListener('pointermove', event => {
    if (!semanticResizer.hasPointerCapture(event.pointerId)) return;
    setSemanticBrowserWidth(resizeStartWidth + event.clientX - resizeStartX);
  });
  semanticResizer.addEventListener('pointerup', event => {
    if (!semanticResizer.hasPointerCapture(event.pointerId)) return;
    semanticResizer.releasePointerCapture(event.pointerId);
    semanticResizer.classList.remove('dragging');
    setSemanticBrowserWidth(semanticBrowser.getBoundingClientRect().width, true);
  });
  semanticResizer.addEventListener('keydown', event => {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    const direction = event.key === 'ArrowLeft' ? -1 : 1;
    setSemanticBrowserWidth(
      semanticBrowser.getBoundingClientRect().width + direction * (event.shiftKey ? 50 : 20),
      true,
    );
    event.preventDefault();
  });
  window.addEventListener('resize', () => {
    if (kind === 'semantic' && window.innerWidth > 850) {
      setSemanticBrowserWidth(semanticBrowser.getBoundingClientRect().width);
    }
  });

  const main = document.querySelector('main');
  const browserPane = document.querySelector('.left');
  const mainResizer = document.querySelector('#main-resizer');
  function setBrowserPaneWidth(width, persist=false) {
    const styles = getComputedStyle(main);
    const available = main.clientWidth
      - parseFloat(styles.paddingLeft) - parseFloat(styles.paddingRight);
    const maximum = Math.max(520, available - 356);
    const next = Math.min(Math.max(520, width), maximum);
    main.style.setProperty('--browser-pane-width', `${next}px`);
    mainResizer.setAttribute('aria-valuenow', String(Math.round(next)));
    mainResizer.setAttribute('aria-valuemax', String(Math.round(maximum)));
    if (persist) {
      try { localStorage.setItem('lexnet-browser-pane-width', String(Math.round(next))); }
      catch (error) { /* Storage may be disabled; resizing still works. */ }
    }
  }

  let savedBrowserWidth = 0;
  try { savedBrowserWidth = Number(localStorage.getItem('lexnet-browser-pane-width')); }
  catch (error) { /* Use the CSS default. */ }
  requestAnimationFrame(() => setBrowserPaneWidth(
    savedBrowserWidth > 0 ? savedBrowserWidth : browserPane.getBoundingClientRect().width,
  ));

  let mainResizeStartX = 0;
  let mainResizeStartWidth = 0;
  mainResizer.addEventListener('pointerdown', event => {
    mainResizeStartX = event.clientX;
    mainResizeStartWidth = browserPane.getBoundingClientRect().width;
    mainResizer.classList.add('dragging');
    mainResizer.setPointerCapture(event.pointerId);
    event.preventDefault();
  });
  mainResizer.addEventListener('pointermove', event => {
    if (!mainResizer.hasPointerCapture(event.pointerId)) return;
    setBrowserPaneWidth(mainResizeStartWidth + event.clientX - mainResizeStartX);
  });
  mainResizer.addEventListener('pointerup', event => {
    if (!mainResizer.hasPointerCapture(event.pointerId)) return;
    mainResizer.releasePointerCapture(event.pointerId);
    mainResizer.classList.remove('dragging');
    setBrowserPaneWidth(browserPane.getBoundingClientRect().width, true);
  });
  mainResizer.addEventListener('keydown', event => {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    const direction = event.key === 'ArrowLeft' ? -1 : 1;
    setBrowserPaneWidth(
      browserPane.getBoundingClientRect().width + direction * (event.shiftKey ? 50 : 20),
      true,
    );
    event.preventDefault();
  });
  window.addEventListener('resize', () => {
    if (window.innerWidth > 850) setBrowserPaneWidth(browserPane.getBoundingClientRect().width);
  });
}
document.querySelectorAll('.tab').forEach(x => x.addEventListener('click', () => selectTab(x.dataset.kind)));
document.querySelectorAll('form').forEach(x => x.addEventListener('submit', search));
renderHead(); initialize();
