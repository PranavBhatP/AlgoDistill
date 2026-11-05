import re

with open('docs/final_presentation.tex', 'r') as f:
    text = f.read()

# Fix bracket issue in Slide 11
text = text.replace(
    r'<|im\_start|>user\\' + '\n' + r'            [ALGORITHM CATEGORY]: dynamic programming\\',
    r'<|im\_start|>user\\' + '\n' + r'            {[ALGORITHM CATEGORY]}: dynamic programming\\'
)

# Replace all \includegraphics with keepaspectratio
def fix_img(m):
    opts = m.group(1)
    filename = m.group(2)
    if 'nitk_logo' in filename:
        return m.group(0) # Keep logo as is
    
    # Determine appropriate height/width
    if filename in ['intro_concept_diagram.png', 'methodology_diagram.png', 'phase5_valley_mapping_pass_at_k.png', 'phase5_category_pass_rates.png']:
        return f'\\includegraphics[width=0.88\\textwidth,height=0.46\\textheight,keepaspectratio]{{{filename}}}'
    elif filename in ['stage1_confusion_matrix.png', 'stage1_per_class_f1.png', 'phase6_repair_pass_at_1_delta_local.png', 'phase6_repair_transitions_local.png']:
        return f'\\includegraphics[width=0.95\\columnwidth,height=0.48\\textheight,keepaspectratio]{{{filename}}}'
    else:
        return f'\\includegraphics[width=0.95\\columnwidth,height=0.58\\textheight,keepaspectratio]{{{filename}}}'

text = re.sub(r'\\includegraphics\[([^\]]+)\]\{([^}]+)\}', fix_img, text)

with open('docs/final_presentation.tex', 'w') as f:
    f.write(text)

print("Applied fixes to docs/final_presentation.tex")
