// One stream reader, one render function per SSE event type.
const $ = (id) => document.getElementById(id);
let papers = [];

const stageNames = ["plan", "papers", "synthesis", "verification"];
const stageMessages = {
  plan: "Planning focused searches…",
  papers: "Searching the academic literature…",
  synthesis: "Connecting the retrieved evidence…",
  verification: "Assessing claims against abstracts…",
};

function progress(stage) {
  const current = stageNames.indexOf(stage);
  for (const [i, name] of stageNames.entries()) {
    const step = document.querySelector(`[data-stage="${name}"]`);
    step.dataset.state = i < current ? "complete" : i === current ? "active" : "waiting";
    step.querySelector(".step-state").textContent = i < current ? "Complete" : i === current ? "In progress" : "Waiting";
    if (i === current) step.setAttribute("aria-current", "step");
    else step.removeAttribute("aria-current");
  }
  $("status").textContent = stageMessages[stage];
}

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    $("question").value = button.dataset.question;
    $("question").focus();
  });
});


async function* readEvents(response) {
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += value.replaceAll("\r\n", "\n");
    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const event = /^event: *(.*)$/m.exec(block)?.[1];
      const data = block.split("\n").filter((l) => l.startsWith("data:")).map((l) => l.slice(5).trimStart()).join("\n");
      if (event) yield { event, data: data ? JSON.parse(data) : {} };
    }
  }
}

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function safeLink(url, label) {
  const a = el("a", label);
  if (/^https?:\/\//i.test(url)) a.href = url;
  a.rel = "noopener noreferrer";
  return a;
}

function fill(id, ...nodes) {
  const section = $(id);
  section.replaceChildren(section.querySelector("h2") ?? [], ...nodes);
  section.hidden = false;
  $("results").hidden = false;
}

const surname = (p) => (p.authors[0]?.trim().split(/\s+/).pop() ?? "").toLowerCase();

function paperFor(citation) {
  const text = citation.toLowerCase();
  const hits = papers.map((p, i) => [p, i]).filter(([p]) => surname(p) && text.includes(surname(p)) && text.includes(String(p.year)));
  return hits.length === 1 ? hits[0][1] : -1;
}

// Inline markdown: **bold**, *italic* and (Author, Year) citations linked to the reference list.
function inline(parent, text, unmatched) {
  const token = /\*\*(.+?)\*\*|\*(.+?)\*|\(([^()]*\d{4}[^()]*)\)/g;
  let last = 0;
  for (const m of text.matchAll(token)) {
    parent.append(text.slice(last, m.index));
    last = m.index + m[0].length;
    if (m[1]) parent.append(el("strong", m[1]));
    else if (m[2]) parent.append(el("em", m[2]));
    else parent.append(citation(m[0], m[3], unmatched));
  }
  parent.append(text.slice(last));
}

function citation(whole, inner, unmatched) {
  const i = paperFor(inner);
  if (i < 0) return el("span", whole, unmatched.includes(inner) || unmatched.includes(whole) ? "cite-unmatched" : "");
  const span = el("span");
  span.append(`(${inner})`);
  const sup = el("sup", undefined, "ref");
  const a = el("a", `[${i + 1}]`);
  a.href = `#paper-${i + 1}`;
  a.setAttribute("aria-label", `Reference ${i + 1}: ${papers[i].title}`);
  sup.append(a);
  span.append(sup);
  return span;
}

function markdown(md, unmatched) {
  const out = document.createDocumentFragment();
  let list = null;
  for (const line of md.split("\n")) {
    const item = /^\s*[-*] +(.*)/.exec(line);
    const heading = /^(#{1,6}) +(.*)/.exec(line);
    if (item) {
      list ??= out.appendChild(el("ul"));
      inline(list.appendChild(el("li")), item[1], unmatched);
      continue;
    }
    list = null;
    if (heading) inline(out.appendChild(el("h3")), heading[2], unmatched);
    else if (line.trim()) inline(out.appendChild(el("p")), line.trim(), unmatched);
  }
  return out;
}

const render = {
  plan(plan) {
    const ol = el("ol");
    for (const q of plan.queries) {
      const li = el("li", q.query);
      li.append(" ", el("span", `(${q.angle})`, "angle"));
      ol.append(li);
    }
    fill("plan", ol);
    progress("papers");
  },
  papers(result) {
    papers = result.papers;
    const nodes = [];
    if (!papers.length) nodes.push(el("p", "No papers found for this question. Try rephrasing it or using more general terms.", "note"));
    const ol = el("ol");
    papers.forEach((p, i) => {
      const li = el("li");
      li.id = `paper-${i + 1}`;
      const names = p.authors.slice(0, 3).join(", ") + (p.authors.length > 3 ? " et al." : "");
      li.append(safeLink(p.doi ? `https://doi.org/${p.doi}` : p.url, p.title), el("div", `${names || "Unknown authors"}, ${p.year ?? "n.d."}`, "meta"));
      ol.append(li);
    });
    nodes.push(ol);
    for (const q of result.failed_queries) nodes.push(el("p", `Search failed for "${q}"; results may be incomplete.`, "warn"));
    fill("papers", ...nodes);
    progress("synthesis");
  },
  synthesis(s) {
    const nodes = [];
    if (s.revision) nodes.push(el("p", "Revised after verification; this replaces the earlier draft.", "note revision-note"));
    else nodes.push(el("p", "Draft, being checked against the abstracts.", "note"));
    nodes.push(markdown(s.markdown, s.unmatched_citations));
    if (s.unmatched_citations.length) nodes.push(el("p", `Cites papers that were not retrieved: ${s.unmatched_citations.join("; ")}`, "warn"));
    fill("synthesis", ...nodes);
    progress("verification");
  },
  verification(v) {
    const labels = { supported: "Supported", partially_supported: "Partially supported", contradicted: "Contradicted", unsupported: "Unsupported" };
    const nodes = [el("p", v.reason, "note")];
    const quotes = (parent, refs) => refs.forEach((r) => {
      const q = el("blockquote", `“${r.quote}” `);
      const a = el("a", `[${r.paper_index}]`);
      a.href = `#paper-${r.paper_index}`;
      q.append(a);
      parent.append(q);
    });
    for (const c of v.claims) {
      const box = el("div", undefined, `claim ${c.assessment}`);
      const claim = el("div");
      inline(claim, c.text, []);
      box.append(el("div", labels[c.assessment], "verdict"), claim, el("div", c.reason, "meta"));
      quotes(box, c.evidence);
      quotes(box, c.counterevidence);
      nodes.push(box);
    }
    for (const f of v.findings) {
      const box = el("div", undefined, "claim contradicted");
      box.append(el("div", f.kind.replace("_", " "), "verdict"), el("div", f.reason));
      quotes(box, f.evidence);
      nodes.push(box);
    }
    fill("verification", ...nodes);
  },
  error(e) {
    $("error").textContent = e.message;
    $("error").hidden = false;
    $("status").textContent = "Research interrupted. Review the error and try again.";
    document.body.dataset.researchState = "error";
    const active = document.querySelector('.steps [data-state="active"]');
    if (active) {
      active.dataset.state = "error";
      active.querySelector(".step-state").textContent = "Interrupted";
      active.removeAttribute("aria-current");
    }
  },
  done() {
    document.body.dataset.researchState = "complete";
    for (const step of document.querySelectorAll(".steps li")) {
      step.dataset.state = "complete";
      step.removeAttribute("aria-current");
      step.querySelector(".step-state").textContent = "Complete";
    }
    $("status").textContent = "Research complete";
    const note = $("synthesis").querySelector(".note:not(.revision-note)");
    if (note) note.textContent = "Read the evidence assessment below before relying on this synthesis.";
  },
};

async function ask(text) {
  const response = await fetch("/api/research", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!response.ok) throw new Error(response.status === 422 ? "Enter a question first." : `The server answered ${response.status}. Try again.`);
  let finished = false;
  for await (const { event, data } of readEvents(response)) {
    render[event]?.(data);
    if (event === "done" || event === "error") finished = true;
  }
  if (!finished) throw new Error("Research stopped before completion. Please try again.");
}

$("ask").addEventListener("submit", async (e) => {
  e.preventDefault();
  const button = e.target.querySelector("button");
  for (const id of ["plan", "papers", "synthesis", "verification", "error"]) $(id).hidden = true;
  papers = [];
  $("results").hidden = true;
  button.disabled = true;
  document.body.dataset.researchState = "running";
  $("ask").setAttribute("aria-busy", "true");
  button.querySelector("span").textContent = "Tracing the evidence…";
  progress("plan");
  try {
    await ask($("question").value.trim());
  } catch (err) {
    render.error({ message: err.message || "Could not reach the server. Check that it is running." });
  } finally {
    button.disabled = false;
    $("ask").removeAttribute("aria-busy");
    button.querySelector("span").textContent = "Trace the evidence";
  }
});
