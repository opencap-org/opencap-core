This document lists the changes to `opencap-core` for each version. When possible, we provide the GitHub issues or pull requests that are related to the items below. If there is no issue or pull request related to the change, then we may provide the commit.

This is not a comprehensive list of changes but rather a hand-curated collection of the more notable ones. For a comprehensive history, see the [OpenCap Core GitHub repo](https://github.com/stanfordnmbl/opencap-core).

v1.1
=====
- Added support for iPhone 17e and iPad Air (A4) (PR #276)
- Improved synchronization with an arm raise (hand punch). (#182)
- Added option for downloading pose pickle files. (#248)
- Moved synchronization specific files to new module utilsSync.py. (PR #259)
- Added main regression tests and sync unit tests. (PR #259)

Previous Changes
================
- 07/03/2024: Speed up IK by removing patella constraints from model ([pull request](https://github.com/stanfordnmbl/opencap-core/pull/174))
- 06/25/2024: Add support for "any pose" scaling ([pull request](https://github.com/stanfordnmbl/opencap-core/pull/168))
- 05/10/2024: Add reprojection error minimization to improve camera synchronization  ([pull request](https://github.com/stanfordnmbl/opencap-core/pull/159))
- 04/05/2024: Update range pelvis translations ([pull request](https://github.com/stanfordnmbl/opencap-core/pull/147))
- 02/23/2024: Add support for setting filter frequency through web app ([pull request](https://github.com/stanfordnmbl/opencap-core/pull/142))
- 07/11/2023: Add support for marker augmenter model v0.3 ([pull request](https://github.com/stanfordnmbl/opencap-core/pull/90))
- 06/09/2023: Add support for full body model with ISB shoulder ([pull request](https://github.com/stanfordnmbl/opencap-core/pull/80))
- 12/07/2022: Add support for iPhone 14, iPhone 14 Plus, iPad (10th gen), iPadPro 12.9 in (6th gen), and iPadPro 11 in (4th gen) ([pull request](https://github.com/stanfordnmbl/opencap-core/pull/17)).
- 11/11/2022: Add support for horizontally oriented videos ([pull request](https://github.com/stanfordnmbl/opencap-core/pull/9)).
- 11/09/2022: Add support for iPhone 14 Pro and iPhone 14 Pro Max ([pull request](https://github.com/stanfordnmbl/opencap-core/pull/4)).
