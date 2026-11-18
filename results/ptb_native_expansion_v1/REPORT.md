# Native measured VCG source audit

Screened 157 source records against 391 target records. Kept 124 source patients, 156 source recordings and 17,573 beats. All 12 shifted/gain-adjusted controls passed. Quarantined 0 source patient IDs with target copy flags or shared exact source recordings.

The public PTB headers define patient-directory membership. The screen checks exact decoded XYZ hashes against every target recording and applies a continuous three-lead correlation check to a beat-template shortlist. Clinical identity and disease labels require different evidence. Training source records use signed mV channels, basic variation and crop checks, and source-only R detection. Original target patient labels, beats and R positions follow the frozen configuration.
