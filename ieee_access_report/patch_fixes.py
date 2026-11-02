import re

with open("ieee_access_report/main_paper.tex", "r") as f:
    text = f.read()

# Fix 1: equation 1 split across lines
old_eq1 = r"""\begin{equation}
\label{eq:valley_base}
\text{Pass@1}_{\text{Base}}: \; 13.2\% \,(10\%) \;\longrightarrow\; \mathbf{11.2\% \,(25\%)} \;\longrightarrow\; 11.6\% \,(50\%) \;\longrightarrow\; 17.6\% \,(100\%)
\end{equation}"""

new_eq1 = r"""\begin{equation}
\label{eq:valley_base}
\begin{aligned}
\text{Pass@1}_{\text{Base}}: \; 13.2\% \,(10\%) &\longrightarrow \mathbf{11.2\% \,(25\%)} \\
&\longrightarrow 11.6\% \,(50\%) \longrightarrow 17.6\% \,(100\%)
\end{aligned}
\end{equation}"""

text = text.replace(old_eq1, new_eq1)

# Fix 2: quote on line 69
text = re.sub(r"describing a [`\\ttex\{\}]+Greedy\'\'[^\n]*?`Dynamic Programming\'\'", 
              r"describing a ``Greedy'' approach in the narrative rationale while implementing a ``Dynamic Programming''", text)

# Fix 3: Figure 10 sizing
text = text.replace(r"\includegraphics[width=0.92\columnwidth]{phase6_repair_pass_at_1_delta_local.png}",
                    r"\vspace{3pt}\includegraphics[width=0.82\columnwidth]{phase6_repair_pass_at_1_delta_local.png}")

# Fix 4: Figure 11 sizing
text = text.replace(r"\includegraphics[width=0.92\columnwidth]{phase6_repair_transitions_local.png}",
                    r"\vspace{3pt}\includegraphics[width=0.85\columnwidth]{phase6_repair_transitions_local.png}")

# Fix 5: Roziere citation
text = re.sub(r"B\.\s*Rozi[^\s]*re", r"B.~Rozi{\\`e}re", text)

with open("ieee_access_report/main_paper.tex", "w") as f:
    f.write(text)

print("Cosmetic fixes successfully applied.")
