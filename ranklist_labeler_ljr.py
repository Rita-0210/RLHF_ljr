# !/usr/bin/env python3
"""
==== No Bugs in code, just some Random Unexpected FEATURES ====
┌─────────────────────────────────────────────────────────────┐
│┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│
││Esc│!1 │@2 │#3 │$4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|\ │`~ ││
│├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│
││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││
│├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│
││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  ││
│├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│
││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││
│└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│
│      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │
│      └───┴─────┴───────────────────────┴─────┴───┘          │
└─────────────────────────────────────────────────────────────┘

Rank List 标注平台，用于标注 Reward Model 的训练数据，通过streamlit搭建。

Author: pankeyu
Date: 2022/1/2
"""
import os
import random

import numpy as np
import pandas as pd
import streamlit as st
#from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
import json


st.set_page_config(
    page_title="Rank List Labeler",
    page_icon='📌',
    layout="wide"
)

MODEL_CONFIG = {
   
    #'dataset_file': 'data/human_labeled/total_dataset.tsv',       # 标注数据集的存放文件
    'q_a_file':'data/q_a.jsonl',
    'rank_list_len': 4,                                           # 排序列表的长度
    #'max_gen_seq_len': 40,                                        # 生成答案最大长度
    # 'random_prompts': [                                           # 随机prompt池
    #                     '今天我去了',
    #                     '这部电影很',
    #                     '刚收到货，感觉',
    #                     '这部电影很',
    #                     '说实话，真的很',
    #                     '这次购物总的来说体验很'
    #                 ],
    'q_a_num' : 3,              
}

######################## 读取jsonl文件 ########################
def load_jsonl(filename):
    with open(filename, "r", encoding='utf-8') as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

def save_jsonl(files, filename):
    with open(filename, "w", encoding='utf-8') as f:
        for line in files:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')


######################## 页面配置初始化 ###########################
RANK_COLOR = [
    'red',
    'green',
    'blue',
    'orange',
    'violet'
]


######################## 会话缓存初始化 ###########################
if 'model_config' not in st.session_state:
    st.session_state['model_config'] = MODEL_CONFIG

# if 'model' not in st.session_state:
#     model_name = st.session_state['model_config']['model_name']
#     st.session_state['model'] = GPT2LMHeadModel.from_pretrained(model_name)

# if 'tokenizer' not in st.session_state:
#     model_name = st.session_state['model_config']['model_name']
#     st.session_state['tokenizer'] = BertTokenizer.from_pretrained(model_name)

# if 'generator' not in st.session_state:
#     st.session_state['generator'] = TextGenerationPipeline(
#         st.session_state['model'],
#         st.session_state['tokenizer'],
#         device=MODEL_CONFIG['device']
#     )

if 'current_results' not in st.session_state:
    st.session_state['current_results'] = [''] * MODEL_CONFIG['rank_list_len']



if 'current_q_a' not in st.session_state:
    st.session_state['current_q_a'] = {"question" : " ","answer" : [" "," "," "," "],"label" : 0}

if 'q_a_list' not in st.session_state:
    st.session_state['q_a_list'] = load_jsonl(MODEL_CONFIG['q_a_file'])

if 'current_prompt' not in st.session_state:
    st.session_state['current_prompt'] = ''

if 'current_idx' not in st.session_state:
    st.session_state['current_idx'] = 0

######################### 函数定义区 ##############################
def generate_text(current_q_a):
    """
    生成对应回答。
    """
    st.session_state['current_results'] = current_q_a["answer"]

# def generate_text():
#     """
#     模型生成文字。
#     """
#     current_results = []
#     for _ in range(MODEL_CONFIG['rank_list_len']):
#         res = st.session_state['generator'](
#                 st.session_state['current_prompt'], 
#                 max_length=MODEL_CONFIG['max_gen_seq_len'], 
#                 do_sample=True
#             )
#         current_results.extend([e['generated_text'] for e in res])
#     st.session_state['current_results'] = current_results

######################### 页面定义区（侧边栏） ########################
st.sidebar.title('📌 人类反馈标注平台')
st.sidebar.markdown('''
    ```python
    用于模型生成回答的标注。
    ```
''')
#st.sidebar.markdown('标注思路参考自 [InstructGPT](https://arxiv.org/pdf/2203.02155.pdf) 。')
#st.sidebar.markdown('RLHF 更多介绍：[想训练ChatGPT？得...](https://zhuanlan.zhihu.com/p/595579042)')
st.sidebar.header('⚙️ 人类反馈标注')
st.sidebar.markdown('给定一个问题，模型将生成4个回答，需要作为眼科医生的你对4个回答进行排序，从1～4，1代表最优回答，次序不可重复，排序完成后点击存储当前排序即可保存')

# st.sidebar.write('当前标注配置（可在源码中修改）：')
# st.sidebar.write(st.session_state['model_config'])

#label_tab, dataset_tab = st.tabs(['Label', 'Dataset'])



######################### 页面定义区（标注页面） ########################
#with label_tab:
with st.expander('🔍 获取问题', expanded=True):
    random_button = st.button('问题', help='从问题库中获取一个问题')

    if random_button:

        for idx in range(MODEL_CONFIG['q_a_num']):
            if "ranked_answer" not in st.session_state['q_a_list'][idx]:
                st.session_state['current_idx'] = idx
                break
        
        current_q_a = st.session_state['q_a_list'][st.session_state['current_idx']]

        prompt_text = current_q_a["question"]

        #prompt_text = random.choice(MODEL_CONFIG['random_prompts'])
    else:
        prompt_text = st.session_state['current_prompt']
        current_q_a = st.session_state['current_q_a']
    
    query_txt = st.text_input('问题: ', prompt_text)

    if current_q_a != st.session_state['current_q_a']:
        st.session_state['current_q_a'] = current_q_a

    if prompt_text != st.session_state['current_prompt']:
        st.session_state['current_prompt'] = prompt_text
        generate_text(current_q_a = current_q_a)

with st.expander('💡 进行排序：为当前问题的回答选择排名，排名数越小，优先值越高，1代表最优先。（-1代表暂未设置排名）', expanded=True):
    if st.session_state['current_results'][0] == '':
        generate_text(current_q_a = current_q_a)

    columns = st.columns([1] * MODEL_CONFIG['rank_list_len'])
    rank_results = [-1] * MODEL_CONFIG['rank_list_len']
    rank_choices = [-1] + [i + 1 for i in range(MODEL_CONFIG['rank_list_len'])]
    for i, c in enumerate(columns):
        with c:
            choice = st.selectbox(f'句子{i+1}排名', rank_choices, help='为当前问题的回答选择排名，排名数越小，优先值越高，1代表最优先。（-1代表当前句子暂未设置排名）')
            if choice != -1 and choice in rank_results:
                st.info(f'当前排名[{choice}]已经被句子[{rank_results.index(choice)+1}]占用，请先将占用排名的句子置为-1再为当前句子分配该排名。')
            else:
                rank_results[i] = choice
            color = RANK_COLOR[i] if i < len(RANK_COLOR) else 'white'
            # st.write(color)
            st.markdown(f":{color}[{st.session_state['current_results'][i]}]")

with st.expander('🥇 排序结果', expanded=True):
    columns = st.columns([1] * MODEL_CONFIG['rank_list_len'])
    for i, c in enumerate(columns):
        with c:
            st.write(f'Rank {i+1}：')
            if i + 1 in rank_results:
                color = RANK_COLOR[rank_results.index(i+1)] if rank_results.index(i+1) < len(RANK_COLOR) else 'white'
                st.markdown(f":{color}[{st.session_state['current_results'][rank_results.index(i+1)]}]")

save_button = st.button('存储当前排序')
if save_button:
    
    if -1 in rank_results:
        st.error('请完成排序后再存储！', icon='🚨')
        st.stop()

    # with open(MODEL_CONFIG['dataset_file'], 'a', encoding='utf8') as f:
    rank_texts = []
    for i in range(len(rank_results)):
        rank_texts.append(st.session_state['current_results'][rank_results.index(i+1)])
    st.session_state['q_a_list'][st.session_state['current_idx']]['ranked_answer'] = rank_texts
    save_jsonl(st.session_state['q_a_list'],MODEL_CONFIG['q_a_file'])
    
    #st.session_state['q_a_list'].remove(st.session_state['current_q_a'])
    if st.session_state['current_idx'] + 1 == MODEL_CONFIG['q_a_num']:
        st.success('已完成所有标注，感谢您的参与！', icon="✅")
    else:
        st.success('保存成功，请点击问题生成新的问答~', icon="✅")

with st.expander('🥇标注题数 ', expanded=True):
    st.markdown(f"当前标注问题数：{st.session_state['current_idx'] + 1}")
    st.markdown(f"剩余问题数：{MODEL_CONFIG['q_a_num'] - st.session_state['current_idx'] - 1}")


######################### 页面定义区（数据集页面） #######################
# with dataset_tab:
#     rank_texts_list = []
#     with open(MODEL_CONFIG['dataset_file'], 'r', encoding='utf8') as f:
#         for i, line in enumerate(f.readlines()):
#             texts = line.strip().split('\t')
#             if len(texts) != MODEL_CONFIG['rank_list_len']:
#                 st.warning(f"error line {i+1}: expeted {MODEL_CONFIG['rank_list_len']} sentence, got {len(texts)}, skipped.")
#                 continue
#             rank_texts_list.append(texts)
#     df = pd.DataFrame(
#         np.array(rank_texts_list),
#         columns=([f'rank {i+1}' for i in range(MODEL_CONFIG['rank_list_len'])])
#     )
#     st.dataframe(df)
