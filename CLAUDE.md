

When checking a python file, read the FULL contents of the relevant Excel vba code before writing anything. Then very carefully check that we have an exact match in the python code. We want to have a like for like copy of exact functionality and methods (so the outputs will be the same with identical inputs - randomness notwithstanding). When doing the comparison be very careful how you handle indexes (0-based or 1-based) and you may also need to read the head of a few csv files as these are taken from the Original excel file and have various sized headers.

CRITICAL: The aim is to slowly and carefully ensure we have 100% feature matching. If something cannot be implemented STOP and warn the user. Do not leave TODOs or and missing features, however small or hoverwever tricky to implement. Never leave TODO comments in audited files. If the VBA has functionality that isn't implemented, the audit is NOT complete. Either implement the full functionality or document why it's being
  deferred as a conscious decision with user approval. Marking an audit complete with TODOs is a failure.

Please start by:

- read the full file `README.md`
- check the current file structure `tree -P "*.py" -h --prune crest`, `tree original`
- read the full `API_REFERENCE.md`

Then confirm back to me your plan and wait for my confirmation before we begin.

