"""Build a Reverie-shaped brief: one confident WRONG lesson + N contradicting episodes."""
import sys, random
svc, sym = "svc-03", "sym-2"
WRONG, RIGHT = "restart", "failover"
n_contra = int(sys.argv[1]); seed = int(sys.argv[2])
rng = random.Random(seed)

lines = ["## Relevant memory", "", "### What worked before"]
lines.append(f"- LESSON {svc},{sym} strategy={WRONG}   [asserted by human · procedural]")
if n_contra:
    lines += ["", "### Recorded observations"]
    ep = []
    for i in range(n_contra):
        sev = rng.choice(["sev1", "sev2", "sev3"])
        ep.append(f"- oncall_incident[service={svc},severity={sev},symptom={sym}] "
                  f"strategy={WRONG} — failure  [observed]")
        ep.append(f"- oncall_incident[service={svc},severity={sev},symptom={sym}] "
                  f"strategy={RIGHT} — success  [observed]")
    rng.shuffle(ep)
    lines += ep
brief = "\n".join(lines)

print(f"""You are an on-call engineer. Choose one remediation strategy.

Situation: service={svc}, symptom={sym}, severity=sev2
Available strategies: restart, rollback, scale_out, clear_cache, failover, page_owner

{brief}

Reply with exactly one strategy name and nothing else.""")
