
We got claude to auto convert all files but it has left many as stubs or put in fake data or not followed the exact functions as the vba code. We are going throguh an audit and fix to address this. 

After that the plan is to run 100 houses and compare to the 100 houses from the excel file in our root dir. We did this for an earlier run and a lot of the scaffolding for setup is done already but should be checked. Note that we are currently reading the "config" from a file - this shouldnt happen we should make the houses stochastically from the total number of houses. 

When checking a python file, read the FULL contents of the relevant Excel vba code before writing anything. Then very carefully check that we have an exact match in the python code. We want to have a like for like copy of exact functionality and methods (so the outputs will be the same with identical inputs - randomness notwithstanding). When doing the comparison be very careful how you handle indexes (0-based or 1-based) and you may also need to read the head of a few csv files as these are taken from the Original excel file and have various sized headers. 

Please start by reading the AUDIT LOG with `head -n 200 AUDIT_LOG.md` to see what the next file to work on is. 

Then confirm back to me your plan and wait for my confirmation before we begin.


