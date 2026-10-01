# Interrupted run — exclude from final comparison

The driver and request harness disappeared after 36/40 first-arm requests,
leaving the separately launched vLLM and LMCache services orphaned. The exact
termination signal was not recorded. No completed comparison or summary exists.
The old tool execution session was no longer available when checked.

All raw data are retained. Restart the full ABBA comparison in v2 under a
detached tmux session with persistent driver logging and a status file.
