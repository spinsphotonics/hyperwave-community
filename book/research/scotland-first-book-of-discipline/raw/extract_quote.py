import re, sys

path = sys.argv[2] if len(sys.argv)>2 else "/home/user/hyperwave-community/book/research/scotland-first-book-of-discipline/text/laing1848-fbd.txt"
raw = open(path, encoding="utf-8").read()

def norm(s):
    return re.sub(r"\s+", " ", s).strip()

ntext = norm(raw)
words = ntext.split(" ")

def find_page(idx_char):
    # idx_char: character index in ntext where quote starts
    # find nearest preceding [[p. N]] marker
    upto = ntext[:idx_char]
    m = list(re.finditer(r"\[\[p\. (\d+)\]\]", upto))
    if m:
        return m[-1].group(1)
    return "?"

def show(phrase, nwords=20):
    nphrase = norm(phrase)
    idx = ntext.find(nphrase)
    if idx == -1:
        print("NOT FOUND:", phrase[:80])
        return
    page = find_page(idx)
    before_text = ntext[:idx]
    after_text = ntext[idx+len(nphrase):]
    before_words = before_text.split(" ")
    # strip trailing page markers tokens like [[p. from before words for cleanliness? keep as is
    before20 = " ".join(before_words[-nwords:])
    after_words = after_text.split(" ")
    after20 = " ".join(after_words[:nwords])
    print("PAGE:", page)
    print("TEXT:", nphrase)
    print("BEFORE20:", before20)
    print("AFTER20:", after20)
    print("---")

if __name__ == "__main__":
    phrase = sys.argv[1]
    show(phrase)
