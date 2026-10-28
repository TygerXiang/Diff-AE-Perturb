#!/usr/bin/env bash
set -u
OUT=/root/perturbench/logs/pkg_probe.log
: > "$OUT"
{
  echo "### import without cwd"
  cd /tmp && python -c "import perturbdiff, sys; print('resolved:', perturbdiff.__file__)"
  echo "### import with cwd=/root/Diff-AE-Perturb"
  cd /root/Diff-AE-Perturb && python -c "import perturbdiff, sys; print('resolved:', perturbdiff.__file__)"
  echo "### pip show"
  python -m pip show perturbdiff 2>&1 | head -12
  echo "### which .pth / easy-install entries mention perturbdiff"
  python - <<'PY'
import site, sys, glob, os
for p in sys.path:
    if p and os.path.isdir(p):
        for f in glob.glob(os.path.join(p, "*.pth")):
            try:
                txt = open(f).read().strip().replace("\n", " | ")
            except Exception as exc:
                txt = f"<{exc}>"
            if "perturbdiff" in txt or "diff-project" in txt or "Diff-AE" in txt:
                print("PTH", f, "->", txt[:200])
PY
} >> "$OUT" 2>&1
echo DONE
