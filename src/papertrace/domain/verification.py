import re
from typing import Literal

from pydantic import BaseModel, Field

from papertrace.domain.citations import citation_candidates
from papertrace.domain.models import Paper, SynthesisResult


class EvidenceReference(BaseModel):
    paper_index: int = Field(ge=1, description="One-based index of the paper in the supplied list.")
    quote: str = Field(
        min_length=1,
        description="The passage of that paper's abstract that backs the point, copied as closely "
        "as you can; the system matches it to the abstract's own sentence and drops it if none "
        "matches.",
    )


class ClaimAssessment(BaseModel):
    text: str = Field(
        min_length=1,
        description="The claim copied exactly from the synthesis, with its attached citation.",
    )
    assessment: Literal[
        "supported", "partially_supported", "contradicted", "unsupported", "unverified"
    ] = Field(
        description="supported: the cited abstracts state it. partially_supported: only part of "
        "it, or a weaker form. contradicted: an abstract states the opposite. unsupported: no "
        "abstract states it, or it is uncited. unverified: you could not find an abstract "
        "passage that backs or contradicts it; use it instead of guessing."
    )
    reason: str = Field(
        min_length=1,
        description="One or two sentences, in the language of the question, justifying the "
        "assessment.",
    )
    evidence: list[EvidenceReference] = Field(
        description="Quotes from the papers cited in this claim that back or limit it; empty "
        "when unsupported."
    )
    counterevidence: list[EvidenceReference] = Field(
        default_factory=list,
        description="Quotes from any retrieved paper that contradict or qualify the claim.",
    )


class VerificationFinding(BaseModel):
    kind: Literal["omission", "contradiction", "ambiguous_citation"] = Field(
        description="omission: a relevant result the synthesis leaves out. contradiction: "
        "abstracts conflict on a point the synthesis presents as settled. ambiguous_citation: "
        "author and year match more than one retrieved paper."
    )
    reason: str = Field(
        min_length=1, description="One or two sentences, in the language of the question."
    )
    evidence: list[EvidenceReference] = Field(
        description="Quotes that back the finding; may be empty only for ambiguous_citation."
    )


class VerificationReport(BaseModel):
    claims: list[ClaimAssessment] = Field(
        description="Every relevant factual claim of the synthesis, uncited ones included."
    )
    findings: list[VerificationFinding] = Field(
        description="Omitted results, contradictions between abstracts and gaps in answering "
        "the question."
    )


class VerificationResult(VerificationReport):
    status: Literal["no_issues_found", "issues_found", "not_verifiable"]
    reason: str
    revision_count: int = Field(default=0, ge=0, le=1)
    # Set by validate_report; internal to the pipeline, so it stays out of the SSE payload.
    needs_revision: bool = Field(default=False, exclude=True)


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_MIN_OVERLAP = 0.7  # share of the quote's words the abstract sentence must contain


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold()))


def locate_quote(quote: str, abstract: str) -> str | None:
    """The text to show as evidence, or None when the abstract does not hold the quote.

    An exact passage is kept. Otherwise (ellipsis, paraphrase, merged sentences) the abstract's
    own best-overlapping sentence replaces it, so the reader only ever sees real abstract text.
    """
    wanted = _words(quote)
    if not wanted:
        return None
    if _normalized(quote) in _normalized(abstract):
        return quote
    sentence = max(_SENTENCE_END.split(abstract), key=lambda s: len(wanted & _words(s)))
    return sentence if len(wanted & _words(sentence)) / len(wanted) >= _MIN_OVERLAP else None


def _grounded(
    refs: list[EvidenceReference], papers: list[Paper], allowed: set[int] | None = None
) -> list[EvidenceReference]:
    """References whose quote really is in the abstract; `allowed` limits which papers count."""
    kept = []
    for ref in refs:
        if ref.paper_index > len(papers) or (
            allowed is not None and ref.paper_index not in allowed
        ):
            continue
        quote = locate_quote(ref.quote, papers[ref.paper_index - 1].abstract or "")
        if quote:
            kept.append(ref.model_copy(update={"quote": quote}))
    return kept


UNVERIFIED_REASON = "No abstract passage could be matched to this claim, so it is not verified."


def validate_report(
    report: VerificationReport, synthesis: SynthesisResult, papers: list[Paper]
) -> VerificationResult:
    """Keep what the abstracts back and say plainly what they do not.

    A claim without checkable evidence becomes `unverified` and an item that cannot be traced
    to the synthesis or an abstract is discarded; neither rejects the whole report.
    """
    claims: list[ClaimAssessment] = []
    discarded = 0
    ambiguous = False
    for claim in report.claims:
        if not _normalized(claim.text) or _normalized(claim.text) not in _normalized(
            synthesis.markdown
        ):
            discarded += 1
            continue
        candidates = citation_candidates(claim.text, papers)
        cited = {i for indices in candidates for i in indices}
        ambiguous = ambiguous or any(len(indices) > 1 for indices in candidates)
        backs = claim.assessment in ("supported", "partially_supported")
        evidence = _grounded(claim.evidence, papers, cited if backs else None)
        counterevidence = _grounded(claim.counterevidence, papers)
        update: dict[str, object] = {"evidence": evidence, "counterevidence": counterevidence}
        if claim.assessment in ("supported", "partially_supported", "contradicted") and not (
            evidence or (claim.assessment == "contradicted" and counterevidence)
        ):
            update |= {"assessment": "unverified", "reason": UNVERIFIED_REASON}
        claims.append(claim.model_copy(update=update))
    findings = []
    for finding in report.findings:
        evidence = _grounded(finding.evidence, papers)
        if evidence or finding.kind == "ambiguous_citation":
            findings.append(finding.model_copy(update={"evidence": evidence}))
        else:
            discarded += 1
    if ambiguous and not any(f.kind == "ambiguous_citation" for f in findings):
        findings.append(
            VerificationFinding(
                kind="ambiguous_citation",
                reason="Author and year identify multiple retrieved papers",
                evidence=[],
            )
        )
    # Ambiguity alone cannot be fixed by rewriting, so it reports an issue without a revision.
    needs_revision = bool(
        synthesis.unmatched_citations
        or any(f.kind != "ambiguous_citation" for f in findings)
        or any(c.assessment != "supported" or c.counterevidence for c in claims)
    )
    status: Literal["no_issues_found", "issues_found", "not_verifiable"] = (
        "issues_found"
        if needs_revision or findings
        else "no_issues_found"
        if claims
        else "not_verifiable"
    )
    unverified = sum(c.assessment == "unverified" for c in claims)
    notes = [
        "Assessment limited to retrieved abstracts" if claims else "No assessed factual claims"
    ]
    if unverified:
        notes.append(
            f"{unverified} claim(s) could not be matched to an abstract and are not verified"
        )
    if discarded:
        notes.append(
            f"{discarded} verifier item(s) discarded: not traceable to the draft or abstracts"
        )
    return VerificationResult(
        claims=claims,
        findings=findings,
        status=status,
        reason=". ".join(notes),
        needs_revision=needs_revision,
    )
