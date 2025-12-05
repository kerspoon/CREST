
Always read the full file `README.md` right now.

We want to have a like for like copy of exact functionality and methods (so the outputs will be the same with identical inputs - randomness notwithstanding). When doing the comparison be very careful how you handle indexes (0-based or 1-based), rounding (Excel's and pythons might be different) and int/float implicit conversion.

CRITICAL: The aim is to slowly and carefully ensure we have 100% feature matching. If something cannot be implemented STOP and warn the user. Do not leave TODOs or and missing features, however small or however tricky to implement. Never leave TODO comments in audited files. If the VBA has functionality that isn't implemented, the audit is NOT complete. Either implement the full functionality or document why it's being deferred as a conscious decision with user approval. 

1. CRITICAL: programme defensively. The user prefers code to crash/exit (with an obvious warning) rather than fail silently. Crashing out is a good thing 0 just stop and let the user know. They will be pleased. 
2. Be concise.
3. Ask the user questions when key decisions need to be made.
4. use `venv/bin/python3` not `python3` or `python`

