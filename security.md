Intended users of the code and/or data of your repo
This code is essentially only intended to be used by myself and my professor.

Assessment of risk of security threats
The most relevant threat here is just the threat of programs being stolen and repurposed by malicious actors, or that these attackers could steal some minor pieces of my identity which are public already. If the data actually did contain sensitive information, that could be a worry as well, but I somewhat doubt that any dataset here would.

Steps I've taken to secure my repo
I set up a new branch ruleset applying to the main branch, which locks down changes to refs as well as requiring pull requests before merging and blocking force pushes. These are all settings that avoid accidental serious changes to the repo, and generally improve the security of it.