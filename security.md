# Repository Security Policy

## 1. Intended Users
The intended users of the code and data in this repository are:
- The repo owner Christopher Frias (Me/ The student)  for development
- Course instructor (Matt the professor) for grading purposes

---

## 2. Risk Assessment of Security Threats
This repository contains academic coursework and implementation exercises. The primary risks if code or data were accessed by unauthorized parties include:
-** Academic integrity violations** are at risks as if someone sees this next year they now have access to completed course material.
-** If this repo contained api keys/database credentials they will be at risk but that is why a .gitignore file exists
- **Data Privacy:** Potential exposure of proprietary datasets or mock personal data used in assignments.
- **Code Tampering / Supply Chain:** Unauthorized modifications that could introduce bugs, grade-impacting regressions, or malicious scripts into automated build/eval workflows.

---

## 3. Steps Taken to Secure the Repository

To mitigate these risks, the following security measures are implemented:

- **Access Controls & Visibility:** 
  - The repository visibility is maintained strictly according to course guidelines (e.g., Private, accessible only by the student and designated course graders).
- **Secret & Credential Management:**
  - A `.gitignore` file is maintained to prevent committing sensitive files (such as `.env`, credentials, local secrets, and build artifacts).
  - Environment variables and secret managers are used for any required API tokens instead of hardcoded strings.
- **Branch Protection & Code Reviews:**
