## Intended Users
This repository is intended for CSE 3000 students and instructors. The code and data are educational materials used to complete homework assignments.

## Security Risk Assessment
Security risk is relatively low because this repository does not contain any confidential information. However, there are some relevant risks if the repository contents were misused.

Malicious changes to the code could introduce vulnerabilities or cause incorrect results in assignments.

Exposing passwords or other sensitive information in this repository could allow unauthorized access to services, so they should not be committed to it.

Because the repository is public, anyone can view and copy its code and data.

## Steps Taken to Secure Repo

The repository contains a CODEOWNERS file to identify the owner for repository changes.

The repository also uses a ruleset, which includes the following security measures:
- Restricting branch deletion.
- Requiring a pull request before merging.
- Requiring at least one approving review before merging.
- Dismissing stale pull request approvals when new commits are pushed.
- Requiring a review from the Code Owner.
- Requiring approval of the most recent reviewable push.
- Requiring conversations resolution before merging.

These protections help ensure that changes to the main branch are reviewed before they are merged, reducing the risk of accidental/unauthorized changes to the repository.