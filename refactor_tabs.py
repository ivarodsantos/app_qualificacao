import os

file_path = "qualificacao_app2.py"

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find the anchor point
anchor_line_idx = -1
for i, line in enumerate(lines):
    if "# (Bloco de conversão redundante removido pois já foi tratado acima)" in line:
        anchor_line_idx = i
        break

if anchor_line_idx == -1:
    print("Anchor not found!")
    exit(1)

# Split content
header_content = lines[:anchor_line_idx]
tab1_content = lines[anchor_line_idx:]

# Define new lines to insert
tabs_def = [
    "\n",
    "tab1, tab2 = st.tabs(['Qualificação Profissional', 'Jornada Empreendedora'])\n",
    "\n",
    "with tab1:\n"
]

# Indent the tab1 content
tab1_content_indented = ["    " + line for line in tab1_content]

# Define tab2 placeholder
tab2_content = [
    "\n",
    "with tab2:\n",
    "    st.header('Jornada Empreendedora (Em Construção)')\n",
    "    st.info('Esta área apresentará os dados de Trilha e Mentoria Empreendedora.')\n"
]

# Write back
new_lines = header_content + tabs_def + tab1_content_indented + tab2_content

with open(file_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Refactor complete.")
