import sys
import re
import os
import csv
import gc
import datetime

import stanza

def file_write(filepath, results):
  if os.path.isfile(filepath):
    print("edit file:" + filepath)
    with open(filepath, mode='a') as f:
        f.writelines(results)
  else:
    print("create file:" + filepath)
    with open(filepath, mode='w') as f:
        f.writelines(results)

#依存関係解析
class DependencyAnalysis:
  # filepath: 読み込みファイルのパス
  # config_gpu: 依存関係解析モデルをGPU上で動かすか否か
  def __init__(self, config_gpu=False):
    self.all_result = []
    # gpuマシンで動かす場合、Falseを変更
    self.nlp = stanza.Pipeline('en', processors='tokenize, mwt, pos, lemma, depparse', use_gpu=config_gpu)

  """
  主節動詞の抽出(非主節の場合は動詞はそもそも抽出されているのでこの関数は実行しない)
  dep: 現在の文章における解析された依存関係
  dependent_root_id: 解析したい依存関係における、より根に近い単語のID
  """
  def extract_verbs(self, dep, dependent_root_id):
    for i in range(len(dep)):
      for d in dep:
        if (d[1] == 'root') or (d[0].id == dependent_root_id):
          if 'VB' in d[2].xpos:
            return d[2].id, '<v> '+d[2].text
          elif dependent_root_id == 0:
            dependent_root_id = d[2].id
    return -1, '<v> <none>'

  """
  入力された動詞idに対する動作主体の抽出。取る可能性のある依存関係はcsubj:~,nsubj:~のみ。
  issue:cc/conjへの対応追加
  引数はextract_verbsに同じ
  """
  # issue:conjunctionの切り分け(単一機能化)
  def extract_subject(self, dep, dependent_root_id):
    for i in dep:
      if ('subj' in i[1]) and (i[0].id == dependent_root_id):
        if 'csubj' in i[1]:
          return i[2].id, ['<cs> '+i[2].text]
        elif 'nsubj' in i[1]:
          cc_result = self.extract_conjunction(dep, i[2].id)
          if cc_result:
            return i[2].id, '<s> '+i[2].text+cc_result
          return i[2].id, '<s> '+i[2].text
    return self.copula_check(dep, dependent_root_id)

  '''
  動作主体が存在しない場合にcopからnsubjが伸びる可能性があるため実施。
  copulaは主語にのみ伸びる可能性があるためobjectは考慮しない．
  詳細はリンクを参照
  https://masayu-a.github.io/UD_Japanese-docs/u-dep/cop.html
  '''
  #4
  def copula_check(self, dep, dependent_root_id):
    for i in dep:
      #copulaをチェックする場合は主語と動詞以外の述語が必ず接続されているため、基本的にはrootに最も近い動詞が参照される
      #copのチェックが必要になるパターンとして節のverbがrootに接続されておらず、adjective等がcopで節のverbに接続されている場合が考えられる。
      #その為copの接続確認はこの他の属性と違って矢印の先が動詞に接続されていることを確認する必要があるためi[2]のidが対象語彙であるか調べる形になっている
      if ('cop' in i[1]) and (i[2].id == dependent_root_id):
        return self.extract_subject(dep, i[0].id)
    return -1, '<s> <none>'

  """
  引数として渡された動詞idに対する動作対象の抽出。取る可能性があるのはobj,obl,ccom系統。
  引数はextract_verbsに同じ
  """
  # issue:conjunctionの切り分け(単一機能化)
  def extract_object(self, dep, dependent_root_id):
    for i in dep:
      if ('obj' in i[1]) and (i[0].id == dependent_root_id):
        cc_result = self.extract_conjunction(dep, i[2].id)
        if cc_result:
          return i[2].id, '<o> '+i[2].text+cc_result
        return i[2].id, '<o> '+i[2].text
      if (('ccomp' in i[1]) or ('xcomp' in i[1])) and (i[0].id == dependent_root_id):
        return i[2].id, ['<co> '+i[2].text]
    return -1, '<o> <none>'

  """
  対象/主体に対する接続詞抽出(conj)
  dep: 現在の文章における解析された依存関係
  dependent_root_id: 接続詞を探索したい単語のid
  """
  def extract_conjunction(self, dep, dependent_root_id):
    for d in dep:
      if ('conj' in d[1]) and (d[0].id == dependent_root_id):
        return self.extract_cc(dep, d[2].id)+d[2].text
    return False


  """
  conjに対するcc(coordination conjunction等位接続詞自身)の抽出
  dep: 現在の文章における解析された依存関係
  dependent_root_id: 接続詞を探索したい単語のid
  """
  def extract_cc(self, dep, dependent_root_id):
    for d in dep:
      if ('cc' == d[1]) and (d[0].id == dependent_root_id):
        return d[2].text + ' '
    return ''


  """
  節ごとの依存関係抽出(途中で目的節、主語節が見つかった場合、先に内部を抽出しに行く)
  dep: 現在の行の依存関係(line_dependencies)
  mcv_id: main clausal verbs id(主節動詞のindex)
  line_result: 現在の行の抽出結果 [verbs, subject, object]
  """
  def extract_clausal_dependencies(self, dep, mcv_id, line_result):
    current_subject_id, current_subject_text = self.extract_subject(dep, mcv_id)
    if type(current_subject_text) == type([]):
      line_result=line_result+' '+self.extract_clausal_dependencies(dep, current_subject_id, current_subject_text[0])
    else:
      line_result=line_result+' '+current_subject_text

    current_object_id, current_object_text = self.extract_object(dep, mcv_id)
    if type(current_object_text) == type([]):
      line_result=line_result+' '+self.extract_clausal_dependencies(dep, current_object_id, current_object_text[0])
    else:
      line_result=line_result+' '+current_object_text
    return line_result

  """
  依存関係の抽出
  line_dependencies:各文章の依存関係木(dict型)
  """
  def extract_dependencies(self, line_dependencies):
    # 探索した依存関係の根側単語id(初期値はrootのため0)
    dependent_root_id = 0
    line_result = ''

    # 主節動詞抽出
    main_clausal_verbs_id, main_clausal_verbs_text = self.extract_verbs(line_dependencies, 0)
    line_result = line_result + main_clausal_verbs_text

    # 主節動詞主体/対象抽出
    line_result = self.extract_clausal_dependencies(line_dependencies, main_clausal_verbs_id, line_result)
    return line_result

  """
  stanzaを用いた文章の解析。解析結果を1行ずつextract_dependsに渡し、依存関係を抽出する
  """
  def sentence_analysis(self, story_sentences):
    line_result = ''
    #EOLの後に存在する(確実ではない)二重スペースを削除，5文章の区切りタグでsplitして1文章ずつの配列に
    story_sentences = story_sentences.replace('  ', '').replace('</s> ', '').replace(" '","'")
    current_depends = self.nlp(story_sentences)
    # 念のため一行ずつの処理(複数文章nlpに渡すとsentencesが長さ1以上のリストになる)
    for idx, analysis_results in enumerate(current_depends.sentences):
      line_result = line_result + self.extract_dependencies(analysis_results.dependencies)
      if idx < len(current_depends.sentences)-1:
        line_result = line_result + ' # '
    return line_result

'''
主節語彙とそれ以外に語彙を分類(解析用)
sequence:'<v> word <s>/<cs> word <o>/<co> word...'(1文章分)
- return
  token_list:主節語彙系列
  s_tokens:非主節主体系列
  o_tokens:非主節対象系列
'''
def clausal_divide(sequence):
  tokens_list = sequence.replace('\n','').replace('  ',' ').split(' ')
  core_clausal_vso = []
  s_tokens = []
  o_tokens = []
  flag = 0
  for idx, token in enumerate(reversed(tokens_list)):
    current_word = []
    if token in '<cs>' or token in '<co>':
      tag = token.replace('c','')
      #cs/coとなっている語彙系列(次の何かしらの特殊トークンまでの語彙系列)をlistにまとめる
      for word in tokens_list[len(tokens_list)-idx:-1]:
        if word in '<':
          break
        current_word.append(word)
      #節を発見した位置からトークン系列末尾まで探索
      for end_idx, current_token in enumerate(tokens_list[len(tokens_list)-idx:-1]):
        if flag == 1:
          s_tokens.append(current_token)
        if flag == 2:
          if '<' in current_token:
            if current_token in '<none>':
              end_idx=end_idx-1
            #節の語彙系列を全て削除，元々のcs/coをs/oに変換
            tokens_list[len(tokens_list)-idx-1] = tag
            del tokens_list[len(tokens_list)-idx+1:len(tokens_list)-idx+end_idx+1]
            flag = 0
            break
          o_tokens.append(current_token)
        if current_token in '<s>':
          flag = 1
        if current_token in '<o>':
          #節の語彙系列を全て削除，元々のcs/coをs/oに変換
          flag = 2
  if tokens_list == ['']:
    tokens_list = ['<v>','<none>','<s>','<none>','<o>','<none>']
  tokens_list = ' '.join(tokens_list).replace('<v> ','').replace(' # ','').split(' <s> ')
  tokens_list = [tokens_list[0]] + tokens_list[1].split(' <o> ')
  return tokens_list, s_tokens, o_tokens

# 実行関数(rocstories)
def run_rocstories(filepath):
  #実行時間表示用
  dt_timedelta = datetime.timedelta(hours=9)
  dt_timezone =datetime.timezone(dt_timedelta, 'JST')
  start_time = format(datetime.datetime.now(dt_timezone),'%Y%m%d_%H%M%S')
  all_result = []
  output_file = filepath+'_extracted_'+ start_time
  
  #ROCStories全てのデータセットに対して抽出を実施
  #まとめて一つのファイルとして解析結果を出力する
  for ext in ['.train','.dev','.test']:
    load_file = open(filepath+ext, 'r')
  da = DependencyAnalysis(config_gpu=True)
  for line_num, current_story in enumerate(load_file):
    if line_num % 500 == 0:
      print('execute: ' + str(line_num) + ' at:' + format(datetime.datetime.now(dt_timezone),'%Y%m%d %H:%M:%S'))
      file_write(output_file, all_result)
      all_result = []
    current_dependencies = ''
    story_title = current_story.split('<EOT>')[0] + '<EOT> '
    #各行をstorylineとsentenceで分割しているタグでsplit，storysentenceを分割して配列として持つ
    story_sentences = current_story.split('<EOL>')[1]
    # 各行の依存関係の抽出
    result = da.sentence_analysis(story_sentences)
    all_result.append(story_title+result+' <EOSC>\n')
  file_write(output_file, all_result)
  load_file.close()
  print('finished at: ' + filepath+ext + ' | time:' + format(datetime.datetime.now(dt_timezone),'%Y%m%d %H:%M:%S'))
  return output_file

# 実行関数(WritingPrompts)
def run_wp(filepath):
  load_file = open(filepath, 'r')
  output_file = filepath+'_extracted'
  all_result = []
  da = DependencyAnalysis(config_gpu=False)
  for line_num, row in enumerate(load_file):
    if line_num % 1000 == 0:
      print('execute: ' + str(line_num))
      file_write(output_file, all_result)
      all_result = []
    current_dependencies = ''
    # 各行の依存関係の抽出
    result = da.sentence_analysis(row)
    all_result.append(result+'<EOSC>\n')
  file_write(output_file, all_result)
  load_file.close()

#データ解析用関数
def analysis_rocstories(result_path, filepath):
  load_file = open(result_path, 'r')
  cmp_rocstories = open(filepath, 'r')
  dt_timedelta = datetime.timedelta(hours=9)
  dt_timezone =datetime.timezone(dt_timedelta, 'JST')
  start_time = format(datetime.datetime.now(dt_timezone),'%Y%m%d_%H%M%S')
  output_file = result_path+'_analysis_'+ start_time
  #vsoの抽出語彙ヒット数(主節vsoを抽出、データセットの要約語彙と比較、当たっていたらhit、外した数はallからhitを引けばよい。allは語彙があったらあっただけカウント。無ければ当然ノーカウント)
  all_vso = [0,0,0]
  hit_vso = [0,0,0]
  all_sl= 0
  all_noncore_so = [0,0]
  hit_noncore_so = [0,0]
  other = 0
  #rocstories(plan-and-write)を1行ずつループ
  for line_num, (row, current_line)  in enumerate(zip(cmp_rocstories,load_file)):
    #各行からタイトルを削除、storylineとstorysentenceを分割して配列として持つ
    slss = row.split('  <EOT> ')[1].split(' <EOL>  </s> ')
    #storyline,storysentence共に扱いやすい形式に変換
    story_lines = slss[0].replace('\t', ' ').split(' # ')

    #実行状況の表示とメモリ解放のためのファイル書き込み
    if line_num % 1000 == 0:
      print('execute: ' + str(line_num) + ' | time:' + str(datetime.datetime.now()))
      all_result = []

    #各行からタイトルを削除、titleとstorylineを分割して配列として持つ
    splited_storylines = current_line.replace('<EOSC>', '').split(' <EOT> ')[1]
    stories_vso = splited_storylines.split(' # ')

    #story_lines(物語文章が1文ずつのstorylineが入った配列)のループ、各文章要約語彙の解析
    for idx, (current_storyline, currentline_vso) in enumerate(zip(story_lines, stories_vso)):
      all_sl = len(current_storyline.split(' '))
      full = full + all_sl
      #現在の比較対象となるstorylineを単語ごとに分割
      storyline_tokens = current_storyline.split(' ')
      main_clausal, s_tokens, o_tokens = clausal_divide(currentline_vso)

      for vso_idx, results_word in enumerate(main_clausal):
        if results_word == '<none>':
          continue
        all_vso[vso_idx] = all_vso[vso_idx] + 1
        for sl_word in storyline_tokens:
          if results_word in sl_word:
            hit_vso[vso_idx] = hit_vso[vso_idx] + 1
            all_sl = all_sl - 1
      for i in s_tokens:
        all_noncore_so[0] =  all_noncore_so[0] + 1
        for j in storyline_tokens:
          if i in j:
            hit_noncore_so[0] = hit_noncore_so[0]+1
            all_sl = all_sl -1
      for i in o_tokens:
        all_noncore_so[1] =  all_noncore_so[1] + 1
        for j in storyline_tokens:
          if i in j:
            hit_noncore_so[1] = hit_noncore_so[1]+1
            all_sl = all_sl -1
      other = other + all_sl
  print(all_vso)
  print(hit_vso)
  print(all_sl)
  print(all_noncore_so)
  print(hit_noncore_so)
  print(other)
  file_write(output_file, all_result)
  load_file.close()
  cmp_rocstories.close()

#各対象ファイルによって処理が少し変化するため、実行関数がいくつか分かれている
if __name__ == '__main__':
  # 既に実行端末でダウンロード済みであれば以下一行は必要なし
  # stanza.download('en')

  #対象ROCStoriesファイルはtitle,storyline,storysentence全て入っているtrain/dev/test
  #train/dev/testそれぞれのデータセットを参照する理由は，拡張子無しのtitlesepkeysepstoryでは何故かstorylineが入っていないため
  #以下リンク参照
  #https://bitbucket.org/VioletPeng/language-model/src/master/rocstory_plan_write/
  filepath = './ROCStories_all_merge_tokenize.titlesepkeysepstory_cmp'
  #抽出用関数
  result_path = run_rocstories(filepath)
  #解析用関数
  analysis_rocstories(result_path, filepath)