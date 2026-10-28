// Run with Playwright installed; PAPERTRACE_URL and PLAYWRIGHT_MODULE can override defaults.
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, "../fixtures/openalex_works.json")));
const work = fixture.results.find(w => w.abstract_inverted_index && Object.keys(w.abstract_inverted_index).length);
const paper = {
  title: work.title, authors: work.authorships.map(a => a.author.display_name),
  year: work.publication_year, doi: work.doi, abstract: null, url: work.id,
};
const positions = Object.entries(work.abstract_inverted_index || {}).flatMap(([word, indexes]) => indexes.map(index => [index, word]));
const quote = positions.sort((a,b)=>a[0]-b[0]).map(entry=>entry[1]).join(" ").slice(0,180);
const claims = ["supported", "partially_supported", "unsupported", "contradicted"].map(assessment => ({
  text: quote, assessment, reason: "Assessment shown for interface validation.",
  evidence: [{paper_index:1, quote}], counterevidence: [],
}));
const out = process.env.PAPERTRACE_SCREENSHOTS || "/tmp/papertrace-ui";
const base = process.env.PAPERTRACE_URL || "http://127.0.0.1:8765";
const ev = (event, data) => "event: " + event + "\ndata: " + JSON.stringify(data) + "\n\n";
const plan = { queries: [{query:"fasting cognition evidence",angle:"cognition"},{query:"fasting clinical outcomes",angle:"clinical evidence"}] };
const synthesis = {
  markdown: "## What the abstracts tell us\nEvidence is limited. (" + paper.authors[0].split(" ").pop() + ", " + paper.year + ")",
  unmatched_citations: [], insufficient_evidence: false,
};
(async () => {
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage();
    const errors = [];
    const failures = [];
    page.on("pageerror", e => errors.push(e.message));
    page.on("console", m => { if (m.type() === "error") errors.push(m.text()); });
    page.on("requestfailed", r => failures.push(r.url()));
    let mode = "partial";
    let requests = 0;
    let release;
    await page.route("**/api/research", async route => {
      requests++;
      if (mode === "loading") await new Promise(resolve => { release = resolve; });
      let body = ev("plan", plan) + (mode === "error"
        ? ev("error", {message:"Semantic Scholar answered HTTP 429. Try again later."})
        : ev("papers", {papers: mode === "empty" ? [] : [paper],failed_queries: mode === "partial" ? ["Semantic Scholar: fasting cognition evidence"] : []})
          + ev("synthesis", {...synthesis, insufficient_evidence: mode === "empty"})
          + ev("verification", {reason:"Abstract-only assessment; inspect the original papers.",claims:mode === "success" ? claims : [],findings:[],revision_count:0,status:"not_verifiable"})
          + (mode === "revision" ? ev("synthesis", {...synthesis,markdown:"Revised evidence summary.",revision:true}) + ev("verification", {reason:"Revised abstract assessment.",claims:[],findings:[],revision_count:1,status:"not_verifiable"}) : "")
          + ev("done", {}));
      if (mode === "interrupted") body = ev("plan", plan);
      await route.fulfill({status:200, contentType:"text/event-stream", body});
    });
    for (const [name,width,height] of [["desktop",1440,1050],["tablet",768,1024],["mobile",390,844],["small",320,812]]) {
      await page.setViewportSize({width,height});
      await page.goto(base);
      assert.equal(await page.locator(".brand img").evaluate(i => i.complete && i.naturalWidth > 0),true);
      await page.screenshot({path:path.join(out, name+"-initial.png"),fullPage:true});
      await page.getByRole("button", {name:"Does intermittent fasting improve memory?"}).click();
      assert.equal(await page.locator("#question").inputValue(),"Does intermittent fasting improve memory?");
      assert.equal(await page.locator("#question").evaluate(e=>document.activeElement===e),true);
      await page.locator("#question").press("Tab");
      await page.keyboard.press("Enter");
      await page.locator("#papers .warn").waitFor();
      await page.getByText("Research complete", {exact:true}).waitFor();
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
      assert.equal(await page.locator("#ask button[type=submit]").isEnabled(),true);
      const reference=page.locator("#synthesis sup a").first();
      await reference.click();
      assert.match(await page.url(),/#paper-1$/);
      await page.screenshot({path:path.join(out,name+"-results.png"),fullPage:true});
    }
    for (mode of ["success","empty","error","revision","loading","interrupted"]) {
      await page.goto(base);
      await page.locator("#question").fill("What do the abstracts say?");
      await page.locator("#ask button[type=submit]").click();
      if (mode === "loading") {
        assert.equal(await page.locator("#ask button[type=submit]").isDisabled(),true);
        await page.screenshot({path:path.join(out,"loading.png"),fullPage:true});
        await page.waitForFunction(()=>document.body.dataset.researchState==="running");
        while (!release) await new Promise(resolve=>setTimeout(resolve,20));
        release();
      }
      await page.locator(["error","interrupted"].includes(mode)?"#error":"#verification").waitFor({state:"visible"});
      if (mode==="revision") {
        assert.match(await page.locator("#synthesis").innerText(),/Revised evidence summary/);
        assert.doesNotMatch(await page.locator("#synthesis").innerText(),/What the abstracts tell us/);
      }
      if (mode==="empty") assert.match(await page.locator("#papers").innerText(),/No papers found/);
      await page.screenshot({path:path.join(out,mode+"-final.png"),fullPage:true});
    }
    mode = "success";
    await page.setViewportSize({width:1440,height:1050});
    await page.goto(base);
    const before = requests;
    await page.locator("#ask button[type=submit]").click();
    assert.equal(await page.locator("#question").evaluate(e=>e.validity.valueMissing),true);
    assert.equal(requests,before);
    await page.locator("#question").fill("Another research question");
    await page.locator("#ask button[type=submit]").click();
    await page.getByText("Research complete",{exact:true}).waitFor();
    assert.equal(await page.locator(".claim").count(),4);
    await page.screenshot({path:path.join(out,"claims-desktop.png"),fullPage:true});
    mode = "error";
    await page.locator("#ask button[type=submit]").click();
    await page.locator("#error").waitFor({state:"visible"});
    assert.equal(await page.locator("#synthesis").isVisible(),false);
    mode = "success";
    await page.locator("#ask button[type=submit]").click();
    await page.getByText("Research complete",{exact:true}).waitFor();
    assert.equal(await page.locator("#error").isVisible(),false);
    await page.emulateMedia({reducedMotion:"reduce",colorScheme:"dark"});
    assert.equal(await page.evaluate(()=>getComputedStyle(document.documentElement).colorScheme),"light");
    assert.deepEqual(errors,[]);
    assert.deepEqual(failures,[]);
    console.log(JSON.stringify({screenshots:out,viewports:4,states:7,errors,failures}));
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exitCode=1;});
