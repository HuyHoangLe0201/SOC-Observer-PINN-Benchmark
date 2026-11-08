ASME JOURNAL OF ELECTROCHEMICAL ENERGY CONVERSION AND STORAGE (JEECS)
SUBMISSION PACKAGE

Manuscript Title:
Comparative Benchmark of State-of-Charge Observers and Physics-Informed Neural Networks Across Battery Aging and Thermal Regimes

Authors:
- Huy Hoang Le (R&D Department, PowerMore Ltd., Da Nang, Vietnam)
- Kim-Anh Nguyen, Ph.D. (Corresponding Author, University of Science and Technology, The University of Danang, Vietnam; email: nkanh@dut.udn.vn)

Contents of this Package:
1. 01_COVER_LETTER_ASME_JEECS.md:
   Formal Cover Letter to Editor-in-Chief Prof. Partha P. Mukherjee (Purdue University).
2. 02_SUBMISSION_METADATA_AND_CHECKLIST.md:
   Complete metadata, 4-bullet highlights, 5 nominated independent international peer reviewers (with contact emails and expertise), CRediT authorship statement, data availability, and 12-page ASME compliance checklist.
3. 03_CAMERA_READY_PDF/:
   - manuscript_jeecs.pdf: Compiled 12-page archival PDF formatted strictly with asmejour.cls (0 Overfull, 0 Underfull, 0 Undefined citations, all 11 figures at full column width).
4. 04_FLAT_SOURCE_FOR_EDITORIAL_MANAGER/:
   - manuscript_jeecs.tex: Self-contained master LaTeX source ready for automated journal build systems.
   - references.bib: BibTeX database with 50 peer-reviewed references with DOIs.
   - asmejour.cls, asmejour.bst, doi.sty, fontaxes.sty, fontaxes-v1.sty: Official ASME journal class and styling dependencies.
   - fig1_*.png to fig11_*.png: All 11 production figures in high resolution (300+ DPI).

Compilation Instructions:
To recompile from 04_FLAT_SOURCE_FOR_EDITORIAL_MANAGER/:
   pdflatex manuscript_jeecs.tex
   bibtex manuscript_jeecs
   pdflatex manuscript_jeecs.tex
   pdflatex manuscript_jeecs.tex
