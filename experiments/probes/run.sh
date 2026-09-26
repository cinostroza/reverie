cd /tmp/probe
: > results.tsv
for n in 0 1 2 4; do
  for seed in 1 2 3 4 5 6 7 8 9 10; do
    (
      out=$(python -X utf8 mkbrief.py "$n" "$seed" | timeout 180 claude -p --model claude-opus-5 2>/dev/null | tr -d '\r' | tr 'A-Z' 'a-z' | grep -oE 'restart|rollback|scale_out|clear_cache|failover|page_owner' | head -1)
      printf '%s\t%s\t%s\n' "$n" "$seed" "${out:-NONE}" >> results.tsv
    ) &
    while [ "$(jobs -r | wc -l)" -ge 5 ]; do wait -n; done
  done
done
wait
echo "DONE $(wc -l < results.tsv) calls"
