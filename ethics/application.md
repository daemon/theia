***A. Describe the purpose (objectives) and rationale of the proposed project and include any hypothesis(es)/research questions to be investigated. This description must make every important point clear without the need to refer to other documents. For a non-clinical study summarize the proposed research using the headings: Purpose, Aim or Hypothesis, and Justification for the Study.   For a clinical trial/medical device testing, summarize the research proposal using the following headings: purpose, hypothesis, justification, and objectives;  See Guidance Note B1a.***

Where available, include a copy of the research proposal as an attachment to this form 101.  A research proposal is required for a clinical trial/medical device testing.

Purpose
Our purpose is to build an application for automating the grading of handwritten math assignments. Specifically, we will examine the task of automatically grading math worksheets from Kumon, a popular learning centre franchise. Upon the successful completion of our study, educators and students could benefit from having an automated grading process. As a side effect, we also will have created a dataset consisting of handwritten digits from children.

Hypothesis
We hypothesize that machine learning techniques can achieve human-level accuracy in the task of grading handwritten Kumon assignments.

Justification for Study
There exist few utilities to perform handwritten examination grading, let alone multiple handwritten expressions consisting of connected digits. Furthermore, existing open datasets for digit recognition contain data from mostly adults. As a result, the datasets provide handwriting that is mostly well-formed and legible. This is often not the case for young children. In order to capture the characteristics of children's handwriting, we need to create our own dataset. If successful, our project will demonstrate that such a utility is feasible, and we will make progress toward deploying a utility for educators and students to use.

---

***B. In LAY LANGUAGE, provide a one paragraph (approximately 100 words) summary of the project including purpose, the anticipated potential benefits, and basic procedures used.***

The purpose of this study is to determine the feasibility of automated grading for handwritten math assignments. We will produce a dataset consisting of digits that are handwritten by children. Upon the conclusion of the study, we will have released the first open dataset that comprises digits written by children. We also anticipate that we could develop an automated grading utility. Our procedure is to collect Kumon worksheets from consenting parents.

---

***B. Provide a detailed, sequential description of the procedures to be used in this study.  For studies involving multiple procedures or sessions, use of a flow chart is expected.  Where applicable, this section also should give the research design (e.g., cross-over design, repeated measures design).***

We will solicit Kumon attendees and their guardians for completed worksheets. The process is to **TODO INFORMATIONAL DIAGRAM**

---

***B. Describe the potential participants in this study including group affiliation, gender, age range and any other special characteristics.  Describe distinct or common characteristics of the potential participants or a group (e.g., a group with a particular health condition) that are relevant to recruitment and/or procedures.   Provide justification for exclusion based on culture, language, gender, race, ethnicity, age or disability.  For example, if a gender or sub-group (i.e., pregnant and/or breastfeeding women) is to be excluded, provide a justification for the exclusion.***

Our target demographics include children and adolescents who attend Kumon, and by extension their parents.

---

***C. How many participants are expected to be involved in this study?  For a clinical trial, medical device testing, or study with procedures that pose greater than minimal risk, sample size determination information is to be provided, as outlined in the Guidance Note C2c.***

We expect to involve 50 participants.

---

***B. Identify who will recruit potential participants and describe the recruitment process. 
Provide a copy of any materials to be used for recruitment (e.g. posters(s), flyers, cards, advertisement(s), letter(s), telephone, email, and other verbal scripts).***

We will coordinate with the owner of a local Kumon centre who will help us identify and recruit subjects.
**TODO**

---

***Will participants receive remuneration (financial, in-kind, or otherwise) for participation?***

They will get to use the mobile application, which saves them time in grading.

---

Describe the plans for provision of study feedback and attach a copy of the feedback letter to be used.  Wherever possible, written feedback should be provided to study participants including a statement of appreciation, details about the purpose and predictions of the study, restatement of the provisions for confidentiality and security of data, an indication of when a study report will be available and how to obtain a copy, contact information for the researchers, and the ethics review and clearance statement. Refer to the sample feedback letters.

**TODO**

---

***1. Identify and describe any known or anticipated direct benefits to the participants from their involvement in the project.  Often there are no direct benefits to participants.  Experiencing an interview, for example, is not a benefit.  Remuneration is not a benefit.***

Participants will receive a graded copy of submitted assignments.

---

***2.Identify and describe any known or anticipated benefits to the scientific community/society from the conduct of this study***

Our study will produce an open dataset of children's handwritten digits.

---

***For each procedure used in this study, provide a description of any known or anticipated risks/stressors to the participants. Consider physiological, psychological, emotional, social, economic risks/stressors. A study-specific current health status form must be included when physiological assessments are used and the associated risk(s) to participants is minimal or greater.***

None

---

***Describe the procedures or safeguards in place to protect the physical and psychological health of the participants in light of the risks/stressors identified in E1.***

None

---

***1. What process will be used to inform the potential participants about the study details and to obtain their consent for participation?***

Information letter with written consent form; provide a hard copy.

If Yes, provide a copy of the Information Letter and Permission Form to be used to obtain permission from those with legal authority to give it. 

**TODO**

---

***1. Provide a detailed explanation of the procedures to be used to ensure anonymity of participants and confidentiality of data both during the research and in the release of the findings.***


In order to train machine learning algorithms, we need to gather photographs of assignments, which will be stored on a backend server once uploaded from the mobile application. Keyed by hash instead of user ID, the stored data will be kept separate from user profiles. Servers and profile information will be protected using best practices (e.g. salted and hashed passwords, diligently patching and updating software, etc.).

The participants will be anonymized by the full removal of their names from any documents procured in the study. If desired by the participant, we will retract and destroy any assignment material deemed relevant, with the exception of handwritten digits.

---

***2. Describe the procedures for securing written records, questionnaires, video/audio tapes and electronic data, etc.  Identify (i) whether the data collected will be linked with any other dataset and identify the linking dataset and (ii) whether the data will be sent outside of the institution where it is collected or if data will be received from other sites.  For the latter, are the data de-identified, anonymized or anonymous; see Guidance Note G***

We will ask for scanned copies and photographs of Kumon math assignments. Where necessary, names of the participants will be removed in the images. The dataset of handwritten digits will not be linked with any other dataset.

