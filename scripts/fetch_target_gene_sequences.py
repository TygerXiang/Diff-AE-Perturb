"""Fetch reference CDS sequences for target genes via NCBI E-utilities.

Writes a two-column TSV (``gene<TAB>sequence``) that
``scripts/export_diffusion_prelim.py --target-sequences`` consumes.

For each gene symbol the script runs esearch against nuccore (human RefSeq
mRNA), then efetch with ``rettype=fasta_cds_na`` so the returned records are
clean nucleotide CDS sequences. When several transcripts exist the longest CDS
is kept, and the accession used for each gene is recorded in a details JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_GENES = [
    "ALDOA", "ATP5E", "BCR", "CAD", "CDC23", "COX11", "DBR1", "DUT",
    "EIF2S1", "GATA1", "GINS1", "GNB2L1", "HSPA5", "HSPA9", "HSPE1",
    "MTOR", "POLR1D", "POLR2H", "RAN", "RPL9", "RPS14", "RPS15",
    "RPS18", "SEC61A1", "TUBB",
]


def fetch(url: str, *, retries: int = 3) -> str:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception as error:  # noqa: BLE001
            last_error = error
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def esearch(gene: str, retmax: int = 10) -> list[str]:
    term = f"{gene}[gene] AND human[orgn] AND refseq[filter] AND biomol_mrna[prop]"
    url = f"{EUTILS}/esearch.fcgi?" + urllib.parse.urlencode(
        {"db": "nuccore", "term": term, "retmode": "json", "retmax": retmax, "sort": "relevance"}
    )
    payload = json.loads(fetch(url))
    return payload.get("esearchresult", {}).get("idlist", [])


def efetch_cds_fasta(ids: list[str]) -> str:
    url = f"{EUTILS}/efetch.fcgi?" + urllib.parse.urlencode(
        {"db": "nuccore", "id": ",".join(ids), "rettype": "fasta_cds_na", "retmode": "text"}
    )
    return fetch(url)


HEADER_RE = re.compile(r"\[gene=([^\]]+)\]")
ACCESSION_RE = re.compile(r"\|([A-Z]{1,2}_[0-9]+\.[0-9]+)_cds_")


def parse_fasta(text: str) -> list[dict]:
    records: list[dict] = []
    header: str | None = None
    chunks: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append({"header": header, "sequence": "".join(chunks)})
            header = line[1:].strip()
            chunks = []
            continue
        if header is not None:
            chunks.append(line.strip())
    if header is not None:
        records.append({"header": header, "sequence": "".join(chunks)})
    for record in records:
        gene_match = HEADER_RE.search(record["header"])
        accession_match = ACCESSION_RE.search(record["header"])
        record["gene"] = gene_match.group(1) if gene_match else ""
        record["transcript"] = accession_match.group(1) if accession_match else ""
        record["sequence"] = re.sub(r"[^ACGTU]", "", record["sequence"].upper()).replace("U", "T")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--genes", nargs="*", default=DEFAULT_GENES)
    parser.add_argument("--details", type=Path, default=None)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    details = []
    lines = []
    for gene in args.genes:
        ids = esearch(gene)
        if not ids:
            details.append({"gene": gene, "status": "no_hit"})
            print(f"{gene}: no hit")
            continue
        records = [record for record in parse_fasta(efetch_cds_fasta(ids)) if record["sequence"]]
        matching = [record for record in records if record["gene"].upper() == gene.upper()]
        pool = matching or records
        if not pool:
            details.append({"gene": gene, "status": "no_cds", "n_records": len(records)})
            print(f"{gene}: no CDS in {len(records)} records")
            continue
        best = max(pool, key=lambda record: len(record["sequence"]))
        lines.append(f"{gene}\t{best['sequence']}")
        details.append(
            {
                "gene": gene,
                "status": "ok",
                "transcript": best["transcript"],
                "header": best["header"],
                "cds_length": len(best["sequence"]),
                "n_candidates": len(pool),
            }
        )
        print(f"{gene}: {best['transcript']} CDS {len(best['sequence'])} nt")
        time.sleep(0.4)

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.details is not None:
        args.details.write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(f"wrote {len(lines)} sequences to {args.output}")


if __name__ == "__main__":
    main()
