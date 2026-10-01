# Do not use this run's deep-KV aggregate as a valid comparison

The three experimental continuation hits were 864, 528, 528 tokens. CPU cache
clear did not invalidate the experimental coarse-prefix index, leaving a stale
longer tail from the previous repetition. The existing runner failed to check
that hit lengths remained consistent; its RESULTS.md aggregate mixes distinct
recovery paths and must not be cited as the deep-KV result.

The native baseline completed independently with three 528-token continuation
hits, ten startup warmups and five successful GPU resets. Its raw data remains
valid and is reused, with explicit provenance, in the corrected v3 experiment.

Fix: ManagementModule.clear now invalidates experimental prefix metadata after
native storage force-clear. A regression test exercises this same management
entry point. The runner now rejects mismatched hit lengths across repetitions.
