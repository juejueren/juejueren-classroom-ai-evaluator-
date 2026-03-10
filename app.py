import streamlit as st
from openai import OpenAI
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import time
import altair as alt
import re
import plotly.graph_objects as go 

# 强制切断代理干扰
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["ALL_PROXY"] = ""

# ==========================================
# 核心配置区
# ==========================================
API_KEY = st.secrets["QWEN_API_KEY"]

# 模块一：课堂调控 Prompt
SYSTEM_PROMPT_M1 = """你是一个严谨的教育评价专家，精通弗兰德斯互动分析与 S-T 课堂分析法。
现在你需要对一段带有时间戳的课堂真实转录文本进行“课堂调控”维度的量化提取。

【核心任务：逐行提取，严禁遗漏】
1. 时间与行为提取：请你**逐行**扫描文本。只要行首出现时间戳（如 "00:00" 或 "20:31-24:00"），就必须提取，**绝对不能跳过任何一个带有时间戳的段落，尤其是学生的发言！**
2. 角色判定：
   - 只要发言人包含“老师”、“教师”等字眼，标记为 "T" (Teacher)。
   - 只要发言人包含“同学”、“学生”、“全体”等字眼，或者是一段纯学生活动/讨论，必须标记为 "S" (Student)。
   - 视频播放或旁白等其他情况，若无明确角色，可根据其实际内容归为 T 或 S，或忽略。
3. 字数统计：
   - 估算教师话语的总字数。
   - 估算教师话语中“疑问句/提问句”的总字数。

【输出格式】
严格输出纯 JSON 对象，不要包含 ```json 等标记符。格式如下：
{
  "segments": [
    {"time": "01:31", "type": "T"},
    {"time": "03:53", "type": "S"},
    {"time": "04:04", "type": "S"}
  ],
  "teacher_total_chars": 500,
  "teacher_question_chars": 120
}
"""

# 模块二：思维激发 Prompt
SYSTEM_PROMPT_M2 = """你是一位资深的思政教育评价专家，精通布鲁姆教育目标分类学与弗兰德斯互动分析。你的任务是对思政课堂实录文本中的“教师话语”进行多维度分析。

# Task
请仔细阅读输入的课堂文本，完成以下两项任务：
1. 提取出教师的所有提问语句（包括设问、反问、启发性疑问等），并进行两个维度的打标：
   - 布鲁姆认知层次分类：低阶问题（记忆、理解、应用）和高阶问题（分析、评价、创造）。
   - “四何”问题分类：知识性问题（是何、为何）和应用性问题（如何、若何）。
2. 估算教师的“间接影响话语”与“直接影响话语”的字数：
   - 间接影响：包括接纳学生感情、表扬或鼓励、采纳学生意见、启发性提问等。
   - 直接影响：包括纯讲授、给出指令、批评或维护权威等。

# Output Format
严格输出一个 JSON 格式，不要包含多余标记：
{
  "questions": [
    {
      "question_text": "提取的教师提问原文",
      "bloom_level": "记忆/理解/应用/分析/评价/创造",
      "four_w": "是何/为何/如何/若何"
    }
  ],
  "teacher_indirect_chars": 200,
  "teacher_direct_chars": 500
}
"""

# 模块三：学科核心素养 Prompt
SYSTEM_PROMPT_M3 = """# Role
你是一位资深的“高中思想政治课教学评价专家”。你的任务是对教师的课堂实录文本进行语义切片，并基于《学科核心素养评价矩阵》对切片进行1-4级的水平打标。

# Task
1. 段落切片：将输入的课堂实录文本，按教师的“讲解/提问/引导”自然语义段落进行切片。一个完整的核心问题或一段连贯的知识点讲解即为一个切片。
2. 维度打标：针对每个切片，分别判断其是否涉及以下四个维度：政治认同、科学精神、法治意识、公共参与。
3. 层级判定：若涉及，请严格按照以下1-4级标准打标；若完全不涉及该维度，则记为 0。

# Evaluation Criteria (1-4级核心内涵)
- 【政治认同】：1级(简单提及素材); 2级(感悟道理/激发自豪); 3级(联系学生生活/自然浸润); 4级(客观呈现成就与挑战/增强使命感)。
- 【科学精神】：1级(单向呈现/封闭提问); 2级(开放问题/有话可说); 3级(按学生学习逻辑递进); 4级(全面呈现事物正反面并分析原因)。
- 【法治意识】：1级(讲授法条/未联系生活); 2级(联系生活/消除距离感); 3级(引导真实行动/怎么做); 4级(培养法治与宪法思维/历史必然性)。
- 【公共参与】：1级(活动流于形式/泛泛而谈); 2级(真实社会调查/直接经验); 3级(设计具体问题解决方案); 4级(价值引导/从我到我们)。

# Output Format
请务必严格输出纯 JSON 格式数据，不要包含任何 Markdown 标记。数据结构如下：
{
  "slices": [
    {
      "slice_id": 1,
      "content": "教师在此处的原话切片...",
      "political_endorsement": 2, 
      "scientific_spirit": 0,
      "rule_of_law": 0,
      "public_participation": 1
    }
  ]
}
"""

# 模块四：评价反馈 Prompt
SYSTEM_PROMPT_M4 = """# Role
你是一位专业的思政课堂教学评估专家，擅长分析课堂实录文本中的师生互动质量与教师评价反馈（理答）的有效性。

# Task
请仔细阅读提供的课堂实录文本，从中提取所有的“学生回答 -> 教师理答”交互对。统计提取的数据，对教师理答的情感倾向进行分类，并输出一段关于本节课“评价反馈”环节的整体定性评价。

# Definitions
1. 学生回答 (Student Answer)：学生在课堂上对教师提问作出的回答或主动发表的见解。
2. 教师理答 (Teacher Feedback)：教师针对前置的“学生回答”做出的明确回应或评价。如果学生回答后，教师无视、未作任何评价或直接进入下一个教学环节，则视为“无明确反馈”。
3. 积极情感 (Positive Sentiment)：教师在理答中使用了表扬、肯定想法、鼓励、赞赏等正向积极的话语（例如：“说得太棒了”、“这个想法很有深度”等）。

# Output Format
请严格以 JSON 格式输出，不要包含任何 Markdown 代码块修饰符（如 ```json），JSON 结构如下：
{
  "interaction_pairs": [
    {
      "student_answer": "学生原话...",
      "teacher_feedback": "教师理答原话...（若无明确反馈，请填 null）",
      "has_feedback": true, 
      "is_positive": true 
    }
  ],
  "qualitative_evaluation": "结合互动情况，生成约 150 字的总体定性评价，说明教师评价的有效性及情感态度。"
}
"""

# ==========================================
# 缓存化的大模型 API 调用函数区 
# ==========================================
@st.cache_data(show_spinner=False, persist="disk")
def fetch_m1_evaluation(text):
    client = OpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    resp = client.chat.completions.create(model="qwen-plus", messages=[{"role": "system", "content": SYSTEM_PROMPT_M1}, {"role": "user", "content": text}], temperature=0.0)
    return resp.choices[0].message.content

@st.cache_data(show_spinner=False, persist="disk")
def fetch_m2_evaluation(text):
    client = OpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    resp = client.chat.completions.create(model="qwen-plus", messages=[{"role": "system", "content": SYSTEM_PROMPT_M2}, {"role": "user", "content": text}], temperature=0.0)
    return resp.choices[0].message.content

@st.cache_data(show_spinner=False, persist="disk")
def fetch_m3_evaluation(text):
    client = OpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    resp = client.chat.completions.create(model="qwen-plus", messages=[{"role": "system", "content": SYSTEM_PROMPT_M3}, {"role": "user", "content": text}], temperature=0.0)
    return resp.choices[0].message.content

@st.cache_data(show_spinner=False, persist="disk")
def fetch_m4_evaluation(text):
    client = OpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    resp = client.chat.completions.create(model="qwen-plus", messages=[{"role": "system", "content": SYSTEM_PROMPT_M4}, {"role": "user", "content": text}], temperature=0.0)
    return resp.choices[0].message.content

def robust_json_parse(text):
    """
    专门用于处理大模型返回的不规则 JSON 字符串的健壮解析器。
    """
    text = text.strip()
    # 1. 剥离 Markdown 格式
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        # 尝试标准解析
        return json.loads(text)
    except json.JSONDecodeError as e:
        # 常见错误修复1：处理大模型在结尾处多加的逗号
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        
        # 常见错误修复2：处理评价文本中未经转义的内部双引号 (非常粗暴但有效的方法)
        # 将内部包含的异常格式替换掉，这里仅做最基本的兜底
        try:
            return json.loads(text)
        except Exception as e2:
            st.warning(f"⚠️ 大模型返回的 JSON 格式存在严重错误，已尝试自动修复但失败。请尝试重新点击运行按钮。错误详情: {e2}")
            # 返回一个空的基础结构，防止后续代码全部崩溃
            return {}
# ==========================================
# 模块计算逻辑区 
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
        range_duration = get_duration_from_range(time_str)
        
        if range_duration is not None: duration = range_duration
        else:
            if i < len(segments) - 1:
                duration = max(1, time_to_sec(segments[i+1].get("time", "00:00")) - curr_sec) 
            else: duration = 30 
        sequence.extend([act_type] * duration)
    return sequence

def calculate_metrics_m1(data):
    if "segments" in data: sequence = expand_time_sequence(data["segments"])
    else: sequence = data.get("sequence", [])
    t_total, t_question = data.get("teacher_total_chars", 1), data.get("teacher_question_chars", 0)
    
    N = len(sequence)
    if N == 0: return None
    NT = sequence.count("T")
    Rt = NT / N
    g = sum(1 for i in range(1, N) if sequence[i] != sequence[i-1]) + 1
    Ch = (g - 1) / N
    TQR = t_question / t_total if t_total > 0 else 0
    
    ch_grade = "A级" if Ch >= 0.4 else "B级" if Ch >= 0.3 else "C级" if Ch >= 0.2 else "D级"
    rt_grade = "A级" if Rt <= 0.5 else "B级" if Rt <= 0.6 else "C级" if Rt <= 0.7 else "D级"
    tqr_grade = "A级" if TQR >= 0.20 else "B级" if TQR >= 0.16 else "C级" if TQR >= 0.10 else "D级"
    t_model = "练习型" if Rt <= 0.3 else "讲授型" if Rt >= 0.7 else "对话型" if Ch >= 0.4 else "混合型"

    return {"N": N, "NT": NT, "Rt": Rt, "Ch": Ch, "TQR": TQR, "sequence": sequence, 
            "grades": {"Ch": ch_grade, "Rt": rt_grade, "TQR": tqr_grade},
            "teaching_model": t_model, "raw_segments": data.get("segments", [])}

def calculate_metrics_m2(ai_data):
    questions = ai_data.get("questions", [])
    bloom_high = sum(1 for q in questions if q.get("bloom_level") in ["分析", "评价", "创造"])
    bloom_low = sum(1 for q in questions if q.get("bloom_level") in ["记忆", "理解", "应用"])
    w_know = sum(1 for q in questions if q.get("four_w") in ["是何", "为何"])
    w_app = sum(1 for q in questions if q.get("four_w") in ["如何", "若何"])

    total_bloom, total_w = bloom_high + bloom_low, w_know + w_app
    high_ratio = bloom_high / total_bloom if total_bloom > 0 else 0
    app_know_ratio = w_app / w_know if w_know > 0 else (1.0 if w_app > 0 else 0)
    app_total_ratio = w_app / total_w if total_w > 0 else 0

    if high_ratio >= 0.5 and app_know_ratio >= 0.666 and app_total_ratio >= 0.4: i_grade = "A级"
    elif high_ratio >= 0.3 and app_know_ratio >= 0.333 and app_total_ratio >= 0.2: i_grade = "B级"
    elif high_ratio >= 0.1 and app_know_ratio >= 0.1 and app_total_ratio >= 0.1: i_grade = "C级"
    else: i_grade = "D级"

    ind_c, dir_c = ai_data.get("teacher_indirect_chars", 0), ai_data.get("teacher_direct_chars", 0)
    id_ratio = ind_c / dir_c if dir_c > 0 else (1.0 if ind_c > 0 else 0.0)
    id_grade = "A级" if id_ratio >= 0.6 else "B级" if id_ratio >= 0.45 else "C级" if id_ratio >= 0.2 else "D级"

    return {"total_q": len(questions), "bloom_high": bloom_high, "bloom_low": bloom_low,
            "w_know": w_know, "w_app": w_app, "high_ratio": high_ratio, "app_know_ratio": app_know_ratio,
            "imagination_grade": i_grade, "raw_questions": questions,
            "indirect_chars": ind_c, "direct_chars": dir_c, "id_ratio": id_ratio, "id_grade": id_grade}

def evaluate_dimension(dimension_key, slices):
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    total_slices = 0
    for s in slices:
        level = s.get(dimension_key, 0)
        if level in counts:
            counts[level] += 1
            total_slices += 1
    n1, n2, n3, n4 = counts[1], counts[2], counts[3], counts[4]
    
    grade = "D级"
    if total_slices > 0:
        hr = (n3 + n4) / total_slices
        lr = n2 / total_slices
        if n4 > 0 and hr >= 0.30: grade = "A级"
        elif (n3 > 0 and hr >= 0.20) or (n4 > 0 and hr < 0.30): grade = "B级"
        elif n3 == 0 and n4 == 0 and lr >= 0.40: grade = "C级"
            
    return {"counts": {"N1": n1, "N2": n2, "N3": n3, "N4": n4}, "total_slices": total_slices, "grade": grade}

def calculate_metrics_m3(ai_data):
    slices = ai_data.get("slices", [])
    if not slices: return None
    return {
        "political_endorsement": evaluate_dimension("political_endorsement", slices),
        "scientific_spirit": evaluate_dimension("scientific_spirit", slices),
        "rule_of_law": evaluate_dimension("rule_of_law", slices),
        "public_participation": evaluate_dimension("public_participation", slices),
        "raw_slices": slices
    }

def calculate_metrics_m4(ai_data):
    pairs = ai_data.get("interaction_pairs", [])
    s_total = len(pairs)
    t_feedback = sum(1 for p in pairs if p.get("has_feedback"))
    t_positive = sum(1 for p in pairs if p.get("is_positive"))
    
    fr = (t_feedback / s_total) if s_total > 0 else 0.0
    pr = (t_positive / t_feedback) if t_feedback > 0 else 0.0
    
    fg = "A级" if fr >= 0.8 else "B级" if fr >= 0.6 else "C级" if fr >= 0.4 else "D级"
    pg = "A级" if pr >= 0.7 else "B级" if pr >= 0.5 else "C级" if pr >= 0.3 else "D级"
    
    return {"S_total": s_total, "T_feedback": t_feedback, "T_positive": t_positive,
            "feedback_rate": fr, "positive_rate": pr, "feedback_grade": fg, "positive_grade": pg,
            "qualitative_evaluation": ai_data.get("qualitative_evaluation", ""), "raw_pairs": pairs}

# ==========================================
# 综合评分计算函数
# ==========================================
def calculate_overall_score(m1, m2, m3, m4):
    def grade_to_score(g_str):
        if not g_str: return 0
        if 'A' in g_str: return 100
        if 'B' in g_str: return 75
        if 'C' in g_str: return 50
        if 'D' in g_str: return 25
        return 0

    bd = {k: {'score': 0, 'weight': 0, 'weighted': 0} for k in 
          ['Ch', 'Rt', 'TQR', 'Img', 'ID', 'Pol', 'Sci', 'Law', 'Pub', 'Fb', 'Pos']}

    if m1:
        for k, v_key in [('Ch', 'Ch'), ('Rt', 'Rt'), ('TQR', 'TQR')]:
            sc = grade_to_score(m1['grades'][v_key])
            bd[k] = {'score': sc, 'weight': 0.05, 'weighted': sc * 0.05}
    
    if m2:
        sc_img, sc_id = grade_to_score(m2['imagination_grade']), grade_to_score(m2['id_grade'])
        bd['Img'] = {'score': sc_img, 'weight': 0.10, 'weighted': sc_img * 0.10}
        bd['ID']  = {'score': sc_id, 'weight': 0.05, 'weighted': sc_id * 0.05}

    if m3:
        for k, v_key in [('Pol', 'political_endorsement'), ('Sci', 'scientific_spirit'), 
                         ('Law', 'rule_of_law'), ('Pub', 'public_participation')]:
            sc = grade_to_score(m3[v_key]['grade'])
            bd[k] = {'score': sc, 'weight': 0.15, 'weighted': sc * 0.15}

    if m4:
        # 修改权重：按照矩阵要求，由于评价有效性和情感共计 0.10
        # 假设反馈率占0.05，情感占0.05
        sc_fb, sc_pos = grade_to_score(m4['feedback_grade']), grade_to_score(m4['positive_grade'])
        bd['Fb']  = {'score': sc_fb, 'weight': 0.05, 'weighted': sc_fb * 0.05}
        bd['Pos'] = {'score': sc_pos, 'weight': 0.05, 'weighted': sc_pos * 0.05}

    total_score = sum([x['weighted'] for x in bd.values()])

    if total_score >= 75: final_grade = "A级"
    elif total_score >= 50: final_grade = "B级"
    elif total_score >= 25: final_grade = "C级"
    else: final_grade = "D级"

    m1_norm = (bd['Ch']['weighted'] + bd['Rt']['weighted'] + bd['TQR']['weighted']) / 0.15 if m1 else 0
    m2_norm = (bd['Img']['weighted'] + bd['ID']['weighted']) / 0.15 if m2 else 0
    m3_norm = sum([bd[k]['weighted'] for k in ['Pol','Sci','Law','Pub']]) / 0.60 if m3 else 0
    m4_norm = (bd['Fb']['weighted'] + bd['Pos']['weighted']) / 0.10 if m4 else 0

    return {"total_score": round(total_score, 1), "final_grade": final_grade, 
            "m1_norm": m1_norm, "m2_norm": m2_norm, "m3_norm": m3_norm, "m4_norm": m4_norm}

# ==========================================
# 可视化图表绘制函数区
# ==========================================
def plot_overall_radar(r_m1, r_m2, r_m3, r_m4):
    categories = ['课堂调控', '思维激发', '核心素养', '评价反馈']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[r_m1, r_m2, r_m3, r_m4, r_m1], theta=categories + [categories[0]],
        fill='toself', fillcolor='rgba(66, 133, 244, 0.3)', line=dict(color='#4285F4', width=2), marker=dict(size=6)
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False,
        margin=dict(t=30, b=30, l=30, r=30), height=300,
        title=dict(text="各维度百分制归一化得分", x=0.5, y=0.98, font=dict(size=14))
    )
    return fig

def plot_st_chart(sequence):
    total_sec = len(sequence)
    interval = 30 if total_sec > 900 else 10
    t_counts, s_counts = [0], [0]
    current_t, current_s = 0, 0
    sampled_seq = sequence[::interval]
    for action in sampled_seq:
        if action == "T": current_t += interval
        elif action == "S": current_s += interval
        t_counts.append(current_t)
        s_counts.append(current_s)
        
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False     
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(t_counts, s_counts, marker='', linestyle='-', color='#1f77b4', linewidth=3, label="实际教学轨迹")
    if len(t_counts) > 0 and len(s_counts) > 0:
        ax.plot(t_counts[-1], s_counts[-1], marker='o', color='red', markersize=8, label="课程结束点")
    
    max_axis = max(2700, total_sec)
    ax.set_xlim(0, max_axis)
    ax.set_ylim(0, max_axis)
    ticks = np.arange(0, max_axis + 1, 300)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{t//60}m" for t in ticks], rotation=45, fontsize=9)
    ax.set_yticklabels([f"{t//60}m" for t in ticks], fontsize=9)
    ax.set_xlabel("教师行为时间 (T)", fontsize=10)
    ax.set_ylabel("学生行为时间 (S)", fontsize=10)
    warning_text = " (⚠️提示:实录时长过短)" if total_sec < 600 else ""
    ax.set_title(f"课堂 S-T 分析图{warning_text}\n(采样间隔: {interval}秒)", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.plot([0, max_axis], [0, max_axis], linestyle='-', color='#FF9999', alpha=0.8, linewidth=1.5, label="均衡线(45°)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    return fig

def plot_rt_ch_chart(Rt, Ch):
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05) 
    ax.set_xlabel("学生课堂主导性 (Rt)", fontsize=10)
    ax.set_ylabel("师生交互有效性 (Ch)", fontsize=10)
    ax.set_title("Rt-Ch 教学模式分析图", fontsize=12, fontweight='bold')
    ax.add_patch(patches.Polygon([(0, 0), (0.3, 0), (0.3, 0.6)], fill=True, color='#d9ead3', alpha=0.9))
    ax.text(0.2, 0.15, '练习型', ha='center', va='center')
    ax.add_patch(patches.Polygon([(0.7, 0), (1, 0), (0.7, 0.6)], fill=True, color='#f4cccc', alpha=0.9))
    ax.text(0.8, 0.15, '讲授型', ha='center', va='center')
    ax.add_patch(patches.Polygon([(0.3, 0), (0.7, 0), (0.7, 0.4), (0.3, 0.4)], fill=True, color='#fff2cc', alpha=0.9))
    ax.text(0.5, 0.2, '混合型', ha='center', va='center')
    ax.add_patch(patches.Polygon([(0.3, 0.4), (0.7, 0.4), (0.7, 0.6), (0.5, 1), (0.3, 0.6)], fill=True, color='#c9daf8', alpha=0.9))
    ax.text(0.5, 0.6, '对话型', ha='center', va='center')
    ax.plot(Rt, Ch, marker='*', color='red', markersize=15)
    return fig

def plot_bloom_pie(high, low):
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'sans-serif']
    fig, ax = plt.subplots(figsize=(4, 4))
    sizes = [high, low] if (high + low) > 0 else [1, 1]
    ax.pie(sizes, labels=['高阶问题', '低阶问题'], colors=['#ff9999', '#66b3ff'], autopct='%1.1f%%', startangle=90)
    ax.axis('equal'); ax.set_title("布鲁姆认知层次分布", fontweight='bold')
    return fig

def plot_four_w_bar(know, app):
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'sans-serif']
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(['知识性\n(是何/为何)', '应用性\n(如何/若何)'], [know, app], color=['#99ff99', '#ffcc99'])
    ax.set_ylabel("问题数量"); ax.set_title("“四何”问题分布图", fontweight='bold')
    return fig

def plot_m3_radar(metrics_m3):
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'sans-serif']
    labels = ['政治认同', '科学精神', '法治意识', '公共参与']
    grade_map = {'A级': 4, 'B级': 3, 'C级': 2, 'D级': 1}
    values = [
        grade_map.get(metrics_m3['political_endorsement']['grade'], 1),
        grade_map.get(metrics_m3['scientific_spirit']['grade'], 1),
        grade_map.get(metrics_m3['rule_of_law']['grade'], 1),
        grade_map.get(metrics_m3['public_participation']['grade'], 1)
    ]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]; angles += angles[:1]
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='#8884d8', alpha=0.35) 
    ax.plot(angles, values, color='#8884d8', linewidth=2.5, marker='o', markersize=8, markerfacecolor='#FFF', markeredgewidth=2)
    ax.set_ylim(0, 4.2); ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(['D', 'C', 'B', 'A'], color='#A0A0A0', size=10)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, size=12, fontweight='bold', color='#333333')
    ax.set_title("核心素养多维均衡度", y=1.1, fontweight='bold', fontsize=14)
    ax.grid(color='#E8E8E8', linestyle='-', linewidth=1)
    ax.spines['polar'].set_color('#D3D3D3')
    return fig

def render_m3_altair_bar(metrics_m3):
    data = []
    dims = [
        ('1_政治认同', metrics_m3['political_endorsement']), ('2_科学精神', metrics_m3['scientific_spirit']),
        ('3_法治意识', metrics_m3['rule_of_law']), ('4_公共参与', metrics_m3['public_participation'])
    ]
    level_names = ['水平1(红:流于形式)', '水平2(黄:浅层联系)', '水平3(蓝:知行转化)', '水平4(绿:价值升华)']
    colors = ['#FF4B4B', '#FACA2B', '#007BFF', '#28A745'] 
    for dim_name, metrics in dims:
        counts = metrics['counts']
        data.extend([
            {"维度": dim_name[2:], "层级": level_names[0], "切片数": counts['N1']},
            {"维度": dim_name[2:], "层级": level_names[1], "切片数": counts['N2']},
            {"维度": dim_name[2:], "层级": level_names[2], "切片数": counts['N3']},
            {"维度": dim_name[2:], "层级": level_names[3], "切片数": counts['N4']}
        ])
    df = pd.DataFrame(data)
    chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X('维度:N', axis=alt.Axis(labelAngle=0, labelFontSize=12), title=None),
        y=alt.Y('切片数:Q', title='切片数量 (个)'),
        color=alt.Color('层级:N', scale=alt.Scale(domain=level_names, range=colors), legend=alt.Legend(title="素养层级判定", orient="bottom")),
        tooltip=['维度', '层级', '切片数']
    ).properties(height=350)
    return chart

# ==========================================
# 网页 UI 设计区
# ==========================================
st.set_page_config(page_title="思政课堂教学评估", layout="wide")
st.title("🎓 思政课堂 AI 教学评估系统")

st.markdown("### 📝 课堂实录输入区")
user_text = st.text_area("请在此粘贴带有时间戳（如 00:00）的真实转录文本，系统将自动进行多维度分析：", height=200)

if st.button("🚀 一键生成课堂多维诊断与综合评分"):
    if not user_text.strip():
        st.warning("请先输入课堂文本哦！")
    else:
        progress_bar = st.progress(0, text="初始化中，准备发送数据...")
        
        try:
            progress_bar.progress(10, text="🧠 正在分析(1/4)：【模块一】课堂调控数据提取中...")
            resp_m1_content = fetch_m1_evaluation(user_text)
            
            progress_bar.progress(30, text="🧠 正在分析(2/4)：【模块二】思维激发与布鲁姆层级判定中...")
            resp_m2_content = fetch_m2_evaluation(user_text)
            
            progress_bar.progress(60, text="🧠 正在分析(3/4)：【模块三】核心素养语义切片中...")
            resp_m3_content = fetch_m3_evaluation(user_text)
            
            progress_bar.progress(80, text="🧠 正在分析(4/4)：【模块四】师生互动与评价反馈分析中...")
            resp_m4_content = fetch_m4_evaluation(user_text)
            
            progress_bar.progress(95, text="✨ AI 分析完毕！正在计算综合得分并生成大屏...")

           ai_data_m1 = robust_json_parse(resp_m1_content)
           ai_data_m2 = robust_json_parse(resp_m2_content)
           ai_data_m3 = robust_json_parse(resp_m3_content)
           ai_data_m4 = robust_json_parse(resp_m4_content)
            
            overall = calculate_overall_score(metrics_m1, metrics_m2, metrics_m3, metrics_m4)
            
            progress_bar.progress(100, text="✅ 评估全面完成！")
            time.sleep(0.5) 
            progress_bar.empty() 

            # ================== 第一部分：总评大屏区 ==================
            st.markdown("---")
            st.markdown("## 🏆 课堂综合评级报告")
            
            col_s1, col_s2 = st.columns([1, 1.5])
            with col_s1:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align: center; border-radius: 10px; padding: 20px; background-color: #f0fdf4;'>"
                            f"<h3 style='color: #2e7d32; margin:0;'>最终得分</h3>"
                            f"<h1 style='color: #4CAF50; font-size: 5rem; margin:0;'>{overall['total_score']}</h1>"
                            f"<h2 style='color: #1b5e20; margin:0;'>评级：{overall['final_grade']}</h2>"
                            f"</div>", unsafe_allow_html=True)
                st.caption("综合评分算法：四模块各项指标等级 A(100) B(75) C(50) D(25) × 评价矩阵对应权重求和。")
            
            with col_s2:
                st.plotly_chart(plot_overall_radar(overall['m1_norm'], overall['m2_norm'], overall['m3_norm'], overall['m4_norm']), use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 📊 详细数据下钻分析")

            # ================== 第二部分：详细模块区 (Tabs) ==================
            tab1, tab2, tab3, tab4 = st.tabs(["📊 模块一：课堂调控", "🧠 模块二：思维激发", "🌟 模块三：核心素养", "🗣️ 模块四：评价反馈"])
            
            with tab1:
                if metrics_m1:
                    st.markdown("#### 🎯 观测点一：师生交互有效性")
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.write("📈 **测量工具**：S-T 图")
                        st.info(f"**师生行为转化率 (Ch)**: {metrics_m1['Ch']:.2f} \n\n等级: **{metrics_m1['grades']['Ch']}**")
                    with col2:
                        st.pyplot(plot_st_chart(metrics_m1['sequence']))
                    st.markdown("---")
                    st.markdown("#### 🎯 观测点二：学生课堂主导性")
                    col3, col4 = st.columns([1, 2])
                    with col3:
                        st.write("🎯 **测量工具**：Rt-Ch 图")
                        st.info(f"**教师行为占有率 (Rt)**: {metrics_m1['Rt']:.2f} \n\n等级: **{metrics_m1['grades']['Rt']}**")
                        st.success(f"💡 **从属诊断：课堂教学模式被判定为【{metrics_m1['teaching_model']}】**")
                    with col4:
                        st.pyplot(plot_rt_ch_chart(metrics_m1['Rt'], metrics_m1['Ch']))
                    st.markdown("---")
                    st.markdown("#### 🎯 观测点三：教师提问频率")
                    st.write("📊 **测量工具**：弗兰德斯互动分析")
                    st.info(f"**教师提问比 (TQR)**: {metrics_m1['TQR']*100:.1f}% \n\n等级: **{metrics_m1['grades']['TQR']}**")
                    with st.expander("🔍 点击查看 AI 提取的时间切片详情"):
                        st.json(metrics_m1['raw_segments'])
                else:
                    st.error("模块一解析失败。")
            
            with tab2:
                if metrics_m2:
                    st.markdown("#### 🎯 观测点一：想象力激发")
                    st.write(f"共识别出 **{metrics_m2['total_q']}** 个有效提问。综合评级：**{metrics_m2['imagination_grade']}**")
                    col1, col2 = st.columns(2)
                    col1.info(f"**高阶问题占比**: {metrics_m2['high_ratio']*100:.1f}%")
                    col2.info(f"**应用性/知识性比值**: {metrics_m2['app_know_ratio']:.2f}")
                    c_chart1, c_chart2 = st.columns(2)
                    with c_chart1: st.pyplot(plot_bloom_pie(metrics_m2['bloom_high'], metrics_m2['bloom_low']))
                    with c_chart2: st.pyplot(plot_four_w_bar(metrics_m2['w_know'], metrics_m2['w_app']))
                    st.markdown("---")
                    st.markdown("#### 🎯 观测点二：教学启发性")
                    st.write("📊 **测量工具**：弗兰德斯互动分析 (I/D比)")
                    st.write(f"教师间接影响话语估算：**{metrics_m2['indirect_chars']}** 字 | 教师直接影响话语估算：**{metrics_m2['direct_chars']}** 字")
                    st.info(f"**教师启发指导比 (I/D)**: {metrics_m2['id_ratio']*100:.1f}%\n\n等级: **{metrics_m2['id_grade']}**")
                    with st.expander("🔍 点击查看 AI 提取的问题语义切片详情"):
                        st.json(metrics_m2['raw_questions'])
                else:
                    st.error("模块二解析失败。")

            with tab3:
                if metrics_m3:
                    st.markdown("### 🌟 综合概览：核心素养多维均衡度")
                    c_chart1, c_chart2 = st.columns([1, 1.5])
                    with c_chart1:
                        st.pyplot(plot_m3_radar(metrics_m3))
                    with c_chart2:
                        st.markdown("**📊 核心素养层级下钻分析 (各水平切片分布)**")
                        st.altair_chart(render_m3_altair_bar(metrics_m3), use_container_width=True)
                    st.markdown("---")
                    
                    def render_m3_observational_point(title, metrics_dict):
                        st.markdown(f"#### 🎯 {title}")
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            st.info(f"**综合评级**: {metrics_dict['grade']} \n\n**有效切片数**: {metrics_dict['total_slices']} 个")
                        with c2:
                            counts = metrics_dict['counts']
                            st.write(f"🟢 **水平4** (价值升华): **{counts['N4']}** 个 \u2003 🟡 **水平3** (知行转化): **{counts['N3']}** 个")
                            st.write(f"🟠 **水平2** (浅层联系): **{counts['N2']}** 个 \u2003 🔴 **水平1** (流于形式): **{counts['N1']}** 个")
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                    render_m3_observational_point("观测点一：政治认同", metrics_m3['political_endorsement'])
                    render_m3_observational_point("观测点二：科学精神", metrics_m3['scientific_spirit'])
                    render_m3_observational_point("观测点三：法治意识", metrics_m3['rule_of_law'])
                    render_m3_observational_point("观测点四：公共参与", metrics_m3['public_participation'])
                    with st.expander("🔍 点击查看 AI 提取的切片原文与打标明细"):
                        st.json(metrics_m3['raw_slices'])
                else:
                    st.error("模块三解析失败。")

            with tab4:
                if metrics_m4:
                    st.markdown("### 🎯 观测点一：课堂评价有效性 (评价反馈率)")
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.info(f"**综合评级**: {metrics_m4['feedback_grade']} \n\n**学生回答总次数**: {metrics_m4['S_total']} 次\n\n**教师有效理答数**: {metrics_m4['T_feedback']} 次")
                        st.write("衡量教师在应对学生表现时，给予明确评价或反馈的次数及占比。")
                    with c2:
                        fig1 = go.Figure(go.Pie(
                            values=[metrics_m4["feedback_rate"], max(0, 1 - metrics_m4["feedback_rate"])],
                            labels=["已理答", "未理答"], hole=0.6, marker_colors=["#4CAF50", "#E0E0E0"], textinfo='none'
                        ))
                        fig1.update_layout(showlegend=True, margin=dict(t=10, b=10, l=10, r=10), height=200,
                                           annotations=[dict(text=f"{metrics_m4['feedback_rate']*100:.1f}%", x=0.5, y=0.5, font_size=24, showarrow=False)])
                        st.plotly_chart(fig1, use_container_width=True)
                        
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("### 🎯 观测点二：评价情感倾向 (积极情感率)")
                    c3, c4 = st.columns([1, 1])
                    with c3:
                        st.info(f"**综合评级**: {metrics_m4['positive_grade']} \n\n**有效理答总数**: {metrics_m4['T_feedback']} 次\n\n**积极情感评价数**: {metrics_m4['T_positive']} 次")
                        st.write("衡量教师的即时评价是否富有启发性且情感丰富。")
                    with c4:
                        fig2 = go.Figure(go.Pie(
                            values=[metrics_m4["positive_rate"], max(0, 1 - metrics_m4["positive_rate"])],
                            labels=["积极情感", "普通/常规"], hole=0.6, marker_colors=["#FF9800", "#E0E0E0"], textinfo='none'
                        ))
                        fig2.update_layout(showlegend=True, margin=dict(t=10, b=10, l=10, r=10), height=200,
                                           annotations=[dict(text=f"{metrics_m4['positive_rate']*100:.1f}%", x=0.5, y=0.5, font_size=24, showarrow=False)])
                        st.plotly_chart(fig2, use_container_width=True)

                    st.markdown("---")
                    st.markdown("### 💡 AI 综合定性评价")
                    st.success(metrics_m4["qualitative_evaluation"])
                    with st.expander("🔍 点击查看 AI 提取的师生交互对明细"):
                        st.json(metrics_m4['raw_pairs'])
                else:
                    st.error("模块四解析失败。")

        except Exception as e:
            progress_bar.empty()
            st.error(f"发生错误：{e}")



