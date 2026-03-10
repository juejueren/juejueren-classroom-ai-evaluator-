import streamlit as st
from openai import OpenAI
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import time
import altair as alt
import re
import plotly.graph_objects as go 
import requests

# ==========================================
# 字体修复：云端中文字体自动配置
# ==========================================
@st.cache_resource
def load_font():
    # 尝试下载文泉驿微米黑字体（在 Streamlit Cloud 上更稳定）
    font_url = "https://github.com/googlefonts/wqy-microhei/raw/main/fonts/wqy-microhei.ttc"
    font_path = "wqy-microhei.ttc"
    if not os.path.exists(font_path):
        try:
            r = requests.get(font_url)
            with open(font_path, "wb") as f:
                f.write(r.content)
        except:
            return None
    return font_path

font_p = load_font()
if font_p:
    zh_font = fm.FontProperties(fname=font_p)
    # 全局注册，防止部分组件依然乱码
    fm.fontManager.addfont(font_p)
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False
else:
    st.warning("中文字体加载失败，图表可能显示乱码。")

# 强制切断代理干扰
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["ALL_PROXY"] = ""

# ==========================================
# 核心配置区
# ==========================================
API_KEY = st.secrets["QWEN_API_KEY"]

# 模块一至四的 Prompt 保持不变 (为了节省篇幅，这里假设已包含你原文中的 SYSTEM_PROMPT_M1...M4)
SYSTEM_PROMPT_M1 = """你是一个严谨的教育评价专家，精通弗兰德斯互动分析与 S-T 课堂分析法...""" # 此处省略你原来的完整Prompt内容
SYSTEM_PROMPT_M2 = """你是一位资深的思政教育评价专家..."""
SYSTEM_PROMPT_M3 = """# Role 你是一位资深的高中思想政治课教学评价专家..."""
SYSTEM_PROMPT_M4 = """# Role 你是一位专业的思政课堂教学评估专家..."""

# ==========================================
# 缓存化的大模型 API 调用函数区 
# ==========================================
@st.cache_data(show_spinner=False, persist="disk")
def fetch_evaluation(text, system_prompt):
    client = OpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    resp = client.chat.completions.create(
        model="qwen-plus", 
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
    )
    return resp.choices[0].message.content

# ==========================================
# 模块计算逻辑区 (保持原逻辑，修复潜在缩进)
# ==========================================
def expand_time_sequence(segments):
    def time_to_sec(t_str):
        try:
            match = re.search(r'(?:(\d+):)?(\d+):(\d+)', str(t_str))
            if match:
                h, m, s = match.groups()
                h = int(h) if h else 0
                return h * 3600 + int(m) * 60 + int(s)
            return 0
        except: return 0
            
    def get_duration_from_range(t_str):
        try:
            matches = re.findall(r'(?:(\d+):)?(\d+):(\d+)', str(t_str))
            if len(matches) == 2:
                s_sec = (int(matches[0][0]) if matches[0][0] else 0) * 3600 + int(matches[0][1]) * 60 + int(matches[0][2])
                e_sec = (int(matches[1][0]) if matches[1][0] else 0) * 3600 + int(matches[1][1]) * 60 + int(matches[1][2])
                return max(1, e_sec - s_sec)
        except: pass
        return None

    sequence = []
    for i in range(len(segments)):
        time_str = segments[i].get("time", "00:00")
        curr_sec = time_to_sec(time_str)
        raw_type = str(segments[i].get("type", "T")).strip().upper()
        act_type = "S" if raw_type.startswith("S") else "T"
        range_duration = get_duration
