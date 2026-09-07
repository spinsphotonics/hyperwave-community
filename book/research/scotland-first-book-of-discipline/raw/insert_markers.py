import re

with open('/home/user/hyperwave-community/book/research/scotland-first-book-of-discipline/raw/fbd_extract_raw.txt', encoding='utf-8') as f:
    lines = f.readlines()

out = []
pat1 = re.compile(r'^\s*(\d{2,3})\s+THE\s+BUKE\s+OF\s+DISCIPLINE\.?\s*$', re.IGNORECASE)
pat2 = re.compile(r'^\s*THE\s+BUKE\s+OF\s+DISCIPLINE\.?\s*(\d{2,3})\s*$', re.IGNORECASE)

count = 0
for line in lines:
    m1 = pat1.match(line)
    m2 = pat2.match(line)
    if m1:
        out.append(f"[[p. {m1.group(1)}]]\n")
        count += 1
        continue
    if m2:
        out.append(f"[[p. {m2.group(1)}]]\n")
        count += 1
        continue
    out.append(line)

print("markers inserted:", count)
with open('/home/user/hyperwave-community/book/research/scotland-first-book-of-discipline/raw/fbd_marked.txt', 'w', encoding='utf-8') as f:
    f.writelines(out)
