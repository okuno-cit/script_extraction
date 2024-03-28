# 概要  
[機械学習を用いた物語生成のための格構造に基づく知識構造抽出手法の提案](https://jglobal.jst.go.jp/detail?JGLOBAL_ID=202302246272973330)、並びに修士論文に関するリポジトリ  
対象とした二つのデータセットに対して文章における主節動詞、主節動詞に対する動作主体、主節動詞に対する動作対象、非主節が抽出可能  

## 対象データセット  
* ROCStoriesデータセット  
[A Corpus and Cloze Evaluation for Deeper Understanding of Commonsense Stories](https://aclanthology.org/N16-1098/)にて提案された、タイトルと5文章からなる大量の物語のデータセット  
[リンク先](https://cs.rochester.edu/nlp/rocstories/)にデータセットについてのページがあり、ページ内メールフォームから著者に連絡をとることでデータセットの取得が可能  
比較対象として使用したものはPlan-and-以下、同データセットをタイトルと物語文章でtab分割したファイルである  

また、当データセットに処理を施したものとして以下リンクの論文が提案されている  
当該論文ではタイトルと文章に分割したデータの文章に対してRakeALgorithmを適用、各文章における要約語彙をストーリーラインとして使用した  
[Plan-And-Write: Towards Better Automatic Storytelling](https://arxiv.org/abs/1811.05701)  
修士論文ではデータ比較においてPlan-and-Writeと同じ並び順のデータが必要であった都合上、下記リンク内titlesepkeysepstoryが拡張子より前のファイル名の末尾につく、train/dev/testの3ファイルを結合したデータを使用して検証を行った  
[bitbucket - データセット](https://bitbucket.org/VioletPeng/language-model/src/master/rocstory_plan_write/)  

* WritingPromptsデータセット  
[Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833)  にて提案された、あるテーマに沿って人間によって記述された約30万の物語のデータセット  
データのソースはRedditのフォーラムであり、フォーラム上で記述された物語をスクレイピングで抽出してきたものをデータセットにしている  
また、[フォーラム](https://www.reddit.com/r/WritingPrompts/)に関しては現在も活動が続いている  
以下リンクから当該データセットはダウンロード可能  
[ダウンロードページ](https://www.kaggle.com/datasets/ratthachat/writing-prompts)  
[タグ付に関する法則(How to Tag Prompts)](https://www.reddit.com/r/WritingPrompts/wiki/how_to_tag_prompts/#wiki_wp.3A_writing_prompt)  


## 対象外データセットを解析する場合  
main.pyにおけるclass DependencyAnalysis内、sentence_analysisに1つの文章または文章を複数含んだ1行を渡すと、stanzaが自身で文章単位に分割、分割結果の文章ごとにextract_dependenciesを実行する  
各行からタイトルを予め抜いておくといった作業は必要だが、確認程度に動かす分には問題なく動作する  

## 各コードについて  
* main.py  
語彙抽出プログラム。  
実行関数、並びにデータ解析用関数は適宜使用するか否か変更する。  

* token_counter.py  
各種データセットにおける文章ごと、物語事の平均抽出語彙数を調べる為のコード  
ownが私の提案手法によって抽出されたデータセットの解析、countが他の論文の手法解析  
