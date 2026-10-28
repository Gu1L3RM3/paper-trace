from typing import Literal

from pydantic import BaseModel, Field

from papertrace.domain.citations import citation_candidates
from papertrace.domain.errors import InvalidReportError
from papertrace.domain.models import Paper, SynthesisResult


class EvidenceReference(BaseModel):
    paper_index: int = Field(ge=1, description="One-based index of the paper in the supplied list.")
    quote: str = Field(
        min_length=1,
        description="One contiguous passage copied exactly from that paper's abstract: no "
        "ellipsis, paraphrase or merged sentences, because it is checked as a literal substring.",
    )


class ClaimAssessment(BaseModel):
    text: str = Field(
        min_length=1,
        description="The claim copied exactly from the synthesis, with its attached citation.",
    )
    assessment: Literal["supported", "partially_supported", "contradicted", "unsupported"] = Field(
        description="supported: the cited abstracts state it. partially_supported: only part of "
        "it, or a weaker form. contradicted: an abstract states the opposite. unsupported: no "
        "abstract states it, or it is uncited."
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


def validate_report(
    report: VerificationReport, synthesis: SynthesisResult, papers: list[Paper]
) -> VerificationResult:
    findings = list(report.findings)
    ambiguous = False
    for claim in report.claims:
        if not _normalized(claim.text) or _normalized(claim.text) not in _normalized(
            synthesis.markdown
        ):
            raise InvalidReportError(
                "Verification claim is absent from the synthesis. "
                "Copy each claim literally from the synthesis"
            )
        candidates = citation_candidates(claim.text, papers)
        cited = {i for indices in candidates for i in indices}
        if claim.assessment in ("supported", "partially_supported"):
            if not claim.evidence or not cited:
                raise InvalidReportError(
                    "Supported claim needs cited abstract evidence. "
                    "Quote the abstract of a paper the claim cites, or mark it unsupported"
                )
            if any(ref.paper_index not in cited for ref in claim.evidence):
                raise InvalidReportError(
                    "Support must come from a cited paper. "
                    "Quote only papers named in the claim's citation; put others in counterevidence"
                )
        if claim.assessment == "contradicted" and not (claim.evidence or claim.counterevidence):
            raise InvalidReportError(
                "Contradicted claim needs evidence. Quote the abstract that states the opposite"
            )
        ambiguous = ambiguous or any(len(indices) > 1 for indices in candidates)
    if ambiguous:
        findings.append(
            VerificationFinding(
                kind="ambiguous_citation",
                reason="Author and year identify multiple retrieved papers",
                evidence=[],
            )
        )
    references = [ref for claim in report.claims for ref in claim.evidence + claim.counterevidence]
    for finding in findings:
        if finding.kind != "ambiguous_citation" and not finding.evidence:
            raise InvalidReportError(
                "Verification finding needs abstract evidence. Add an exact abstract quote for it"
            )
        references.extend(finding.evidence)
    for ref in references:
        if ref.paper_index > len(papers):
            raise InvalidReportError(
                "Verification references an unknown paper. "
                "Use only the one-based indices of the supplied papers"
            )
        abstract = papers[ref.paper_index - 1].abstract or ""
        if not _normalized(ref.quote) or _normalized(ref.quote) not in _normalized(abstract):
            raise InvalidReportError(
                "Verification quote is absent from the abstract. "
                "Quote one contiguous passage copied exactly"
            )
    # Ambiguity alone cannot be fixed by rewriting, so it reports an issue without a revision.
    needs_revision = bool(
        synthesis.unmatched_citations
        or any(f.kind != "ambiguous_citation" for f in findings)
        or any(c.assessment != "supported" or c.counterevidence for c in report.claims)
    )
    status: Literal["no_issues_found", "issues_found", "not_verifiable"] = (
        "issues_found"
        if needs_revision or findings
        else "no_issues_found"
        if report.claims
        else "not_verifiable"
    )
    return VerificationResult(
        claims=report.claims,
        findings=findings,
        status=status,
        reason="Assessment limited to retrieved abstracts"
        if report.claims or status == "issues_found"
        else "No assessed factual claims",
        needs_revision=needs_revision,
    )
