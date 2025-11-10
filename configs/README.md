# Dwelling Configurations

## ⚠️ TEMPORARY - For Validation Only

This directory contains dwelling configurations extracted from the Excel baseline file for validation purposes.

**Current Status:**
- `excel_100houses.json` - Extracted from `CREST_Demand_Model_v2.3.3__100_houses.xlsm`
- Used with `--config-file` flag to run identical dwelling setups as Excel

**Future Plan:**
Replace config file loading with **stochastic dwelling generation** matching Excel VBA logic:
- Random number of residents (1-6) based on UK household distribution
- Random building types based on building stock probabilities
- Random heating systems based on ownership statistics
- Random PV/solar thermal based on penetration rates

This will allow truly stochastic simulations instead of pre-configured dwelling lists.
