from typing import Literal

from pydantic import BaseModel, Field

from papertrace.domain.citations import citation_candidates
from papertrace.domain.errors import InvalidReportError
from papertrace.domain.models import Paper, SynthesisResult


class EvidenceReference(BaseModel):
    paper_index: int = Field(ge=1)
    quote: str = Field(min_length=1)


class ClaimAssessment(BaseModel):
    text: str = Field(min_length=1)
    assessment: Literal["supported", "partially_supported", "contradicted", "unsupported"]
    reason: str = Field(min_length=1)
    evidence: list[EvidenceReference]
    counterevidence: list[EvidenceReference] = Field(default_factory=list)


class VerificationFinding(BaseModel):
    kind: Literal["omission", "contradiction", "ambiguous_citation"]
    reason: str = Field(min_length=1)
    evidence: list[EvidenceReference]


class VerificationReport(BaseModel):
    claims: list[ClaimAssessment]
    findings: list[VerificationFinding]


class VerificationResult(VerificationReport):
    status: Literal["no_issues_found", "issues_found", "not_verifiable"]
    reason: str
    revision_count: int = Field(default=0, ge=0, le=1)


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def validate_report(
    report: VerificationReport, synthesis: SynthesisResult, papers: list[Paper]
) -> VerificationResult:
    findings = list(report.findings)
    for claim in report.claims:
        if not _normalized(claim.text) or _normalized(claim.text) not in _normalized(
            synthesis.markdown
        ):
            raise InvalidReportError("Verification claim is absent from the synthesis")
        candidates = citation_candidates(claim.text, papers)
        cited = {i for indices in candidates for i in indices}
        if claim.assessment in ("supported", "partially_supported"):
            if not claim.evidence or not cited:
                raise InvalidReportError("Supported claim needs cited abstract evidence")
            if any(ref.paper_index not in cited for ref in claim.evidence):
                raise InvalidReportError("Support must come from a cited paper")
        if claim.assessment == "contradicted" and not (claim.evidence or claim.counterevidence):
            raise InvalidReportError("Contradicted claim needs evidence")
        if any(len(indices) > 1 for indices in candidates):
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
            raise InvalidReportError("Verification finding needs abstract evidence")
        references.extend(finding.evidence)
    for ref in references:
        if ref.paper_index > len(papers):
            raise InvalidReportError("Verification references an unknown paper")
        abstract = papers[ref.paper_index - 1].abstract or ""
        if not _normalized(ref.quote) or _normalized(ref.quote) not in _normalized(abstract):
            raise InvalidReportError("Verification quote is absent from the abstract")
    issues = bool(
        findings
        or synthesis.unmatched_citations
        or any(c.assessment != "supported" or c.counterevidence for c in report.claims)
    )
    status: Literal["no_issues_found", "issues_found", "not_verifiable"] = (
        "issues_found" if issues else "no_issues_found"
    )
    if not report.claims:
        status = "not_verifiable"
    return VerificationResult(
        claims=report.claims,
        findings=findings,
        status=status,
        reason="No assessed factual claims"
        if not report.claims
        else "Assessment limited to retrieved abstracts",
    )
