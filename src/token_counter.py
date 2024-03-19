import os

from torchtext import data

def countup(filepath):
  f = open(filepath)
  lines = f.readlines()
  len_tag = 0
  len_notag = 0
  sentences_count = 0
  for i in lines:
    title_split = i.split("<EOT> ")
    #sl_split = title_split[1].split("\t")
    #tokens = sl_split[0].replace(' #','').replace(' <EOL>','').split(" ")
    tokens = i.replace(' #','').replace(' <EOL>','').split(" ")
    len_notag = len_notag + len(tokens)
    #len_tag = len_tag + len(sl_split[0].split(' '))
    len_tag = len_tag + len(i.split(' '))
    sentences_count = sentences_count + i.count('#') + 1
  return len_tag, len_notag, len(lines), sentences_count

def countup_own(filepath):
  f = open(filepath)
  lines = f.readlines()
  len_notag = 0
  len_tag = 0
  sentences_count = 0
  for i in lines:
    title_split = i.split("<EOT> ")
    tokens = title_split[1].replace('<v> ','').replace('<s> ','').replace('<o> ','').replace('<cs> ','').replace('<co> ','').replace('# ','').replace(' <EOSC>','').replace(' <EOL>', '').split(" ")
    len_notag = len_notag + len(tokens)
    len_tag = len_tag + len(title_split[1].split(' '))
    sentences_count = sentences_count + i.count('#') + 1
  return len_tag, len_notag, len(lines), sentences_count

filepath = './ROCStories_all_merge_tokenize.titlesepkeysepstory_extracted_20240128_094517'
filepath = './train.wp_target_extracted20240128_221627'
result_len_tag, result_len_notag, result_line_count, sentences_count = countup(filepath)
#result_len_tag, result_len_notag, result_line_count, sentences_count = countup_own(filepath)

TEXT = data.Field(sequential=True, include_lengths=True)
LABEL = data.LabelField()
train = data.TabularDataset(path=filepath, format='tsv', fields=[('text', TEXT), ('label', LABEL)], skip_header=True)
TEXT.build_vocab(train)

# <unk>/<pad>未使用のため-2
print(len(TEXT.vocab.itos)-2)

print("avg tag: "+str(result_len_tag/result_line_count))
print("avg notag: "+str(result_len_notag/result_line_count))
print("line count: "+str(result_line_count))
print("sentence count: "+str(sentences_count))