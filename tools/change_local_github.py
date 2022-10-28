file = './pages/Pages04-ClassicalAlgorithm.html'

with open(file, 'r', encoding='utf8') as f:
    text = f.read()

import re 
nodes_a = re.findall(r'<a class="ipynb2">.*?</a>', text)
names_a = re.findall(r'<a class="ipynb2"><span>(.*?)</span></a>', text)

prefix = 'https://github.com/lizidoufu/lizidoufu.github.io/tree/master/ipynb2/history/'
links_a = ['<a class="ipynb2" href=\"' + prefix + i + '\"' + " target=\"_blank\" style=\"text-decoration:none;\">" for i in names_a]


for i in range(len(nodes_a)):
    node = nodes_a[i]
    link = links_a[i]
    newd = node.replace("<a class=\"ipynb2\">", link)
    text = re.sub(node, newd, text)

with open(file, 'w', encoding="utf8") as f:
    f.write(text)