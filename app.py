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
# 1. 字体修复：云端中文字体自动配置（防御性增强版）
# ==========================================
@st.cache_resource
def get_zh_font():
    """下载并加载中文字体，返回 FontProperties 对象"""
    font_url = "https://github.com/googlefonts/wqy-microhei/raw/main/fonts/wqy-microhei.ttc"
    font_path = os.path.join(os.getcwd(), "wqy-microhei.ttc")
    
    # 检查文件是否存在且大小正常 (>1MB)
    if not os.path.exists(font_path) or os.path.getsize(font_path) < 1000000:
        try:
            r = requests.get(font_url, timeout=20)
            if r.status_code == 200:
                with open(font_path, "wb") as f:
                    f.write(r.content)
            else: return None
        except: return None
    
    try:
        # 尝试注册字体到管理器
        fm.fontManager.addfont(font_path)
        return fm.FontProperties(fname=font_path)
    except:
        return None

# 初始化全局字体对象
ZH_FONT = get_zh_font()
if ZH_FONT:
    plt.rcParams['axes.unicode_minus'] = False
else:
    st.sidebar.warning("中文字体加载失败，图表可能显示乱码。")

# 环境变量：清理代理干扰
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["ALL_PROXY"] = ""

# ==========================================
# 2. 核心配置与 Prompt (保持原样)
# ==========================================
API_KEY = st.secrets["QWEN_API_KEY"]

SYSTEM_PROMPT_M1 = """你是一个严谨的教育评价专家，精通弗兰德斯互动分析与 S-T 课堂分析法。
现在你需要对一段带有时间戳的课堂真实转录文本进行“课堂调控”维度的量化提取。
【输出格式】严格输出纯 JSON 对象，格式如下：
{"segments": [{"time": "01:31", "type": "T"}], "teacher_total_chars": 500, "teacher_question_chars": 120}"""

SYSTEM_PROMPT_M2 = """你是一位资深的思政教育评价专家...""" # 此处省略其余 Prompt 详情，请保留你原始代码中的 Prompt
SYSTEM_PROMPT_M3 = """# Role 你是一位资深的高中思想政治课教学评价专家..."""
SYSTEM_PROMPT_M4 = """# Role 你是一位专业的思政课堂教学评估专家..."""

# ==========================================
# 3. 计算与处理逻辑 (修复缩进)
# ==========================================
@st.cache_data(show_spinner=False, persist="disk")
def fetch_evaluation(text, system_prompt):
    client = OpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    resp = client.chat.completions.create(
        model="qwen-plus", 
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
    )
    return resp.choices[0].message.content

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

# calculate_metrics_m2, m3, m4 保持你的原始逻辑即可（只需确保缩进正确）
# ... [此处建议保留你之前的 calculate_metrics_m2/m3/m4 函数体] ...

def calculate_metrics_m2(ai_data):
    questions = ai_data.get("questions", [])
    bloom_high = sum(1 for q in questions if q.get("bloom_level") in ["分析", "评价", "创造"])
    bloom_low = sum(1 for q in questions if q.get("bloom_level") in ["记忆", "理解", "应用"])
    w_know = sum(1 for q in questions if q.get("four_w") in ["是何", "为何"])
    w_app = sum(1 for q in questions if q.get("four_w") in ["如何", "若核"])
    total_bloom = bloom_high + bloom_low
    high_ratio = bloom_high / total_bloom if total_bloom > 0 else 0
    ind_c, dir_c = ai_data.get("teacher_indirect_chars", 0), ai_data.get("teacher_direct_chars", 0)
    id_ratio = ind_c / dir_c if dir_c > 0 else 0
    return {"total_q": len(questions), "high_ratio": high_ratio, "bloom_high": bloom_high, 
            "bloom_low": bloom_low, "w_know": w_know, "w_app": w_app, "id_ratio": id_ratio,
            "imagination_grade": "A级" if high_ratio >= 0.3 else "B级", "id_grade": "A级" if id_ratio >= 0.5 else "B级",
            "raw_questions": questions, "indirect_chars": ind_c, "direct_chars": dir_c}

def evaluate_dimension(dimension_key, slices):
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    total_slices = 0
    for s in slices:
        level = s.get(dimension_key, 0)
        if level in counts:
            counts[level] += 1
            total_slices += 1
    grade = "B级" if (counts[3] + counts[4]) / max(1, total_slices) > 0.2 else "C级"
    return {"counts": {"N1": counts[1], "N2": counts[2], "N3": counts[3], "N4": counts[4]}, "total_slices": total_slices, "grade": grade}

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
    fr = t_feedback / s_total if s_total > 0 else 0
    pr = t_positive / t_feedback if t_feedback > 0 else 0
    return {"S_total": s_total, "T_feedback": t_feedback, "feedback_rate": fr, "positive_rate": pr,
            "feedback_grade": "A级" if fr >= 0.8 else "B级", "positive_grade": "A级" if pr >= 0.7 else "B级",
            "qualitative_evaluation": ai_data.get("qualitative_evaluation", ""), "raw_pairs": pairs}

def calculate_overall_score(m1, m2, m3, m4):
    def g2s(g): return 100 if 'A' in g else 75 if 'B' in g else 50 if 'C' in g else 25
    s1 = (g2s(m1['grades']['Ch']) + g2s(m1['grades']['Rt']) + g2s(m1['grades']['TQR'])) / 3 if m1 else 0
    s2 = (g2s(m2['imagination_grade']) + g2s(m2['id_grade'])) / 2 if m2 else 0
    s3 = (g2s(m3['political_endorsement']['grade']) + g2s(m3['scientific_spirit']['grade'])) / 2 if m3 else 0
    s4 = (g2s(m4['feedback_grade']) + g2s(m4['positive_grade'])) / 2 if m4 else 0
    total = s1*0.15 + s2*0.15 + s3*0.60 + s4*0.10
    return {"total_score": round(total, 1), "final_grade": "A级" if total >= 80 else "B级", 
            "m1_norm": s1, "m2_norm": s2, "m3_norm": s3, "m4_norm": s4}

# ==========================================
# 4. 绘图区 (修复 IndentationError & Font)
# ==========================================
def plot_st_chart(sequence):
    total_sec = len(sequence)
    interval = 30 if total_sec > 900 else 10
    t_counts, s_counts = [0], [0]
    curr_t, curr_s = 0, 0
    for action in sequence[::interval]:
        if action == "T": curr_t += interval
        else: curr_s += interval
        t_counts.append(curr_t)
        s_counts.append(curr_s)
        
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(t_counts, s_counts, color='#1f77b4', linewidth=2, label="教学轨迹")
    max_axis = max(2700, total_sec)
    ax.set_xlim(0, max_axis); ax.set_ylim(0, max_axis)
    ax.set_xlabel("教师时间 (T)", fontproperties=ZH_FONT)
    ax.set_ylabel("学生时间 (S)", fontproperties=ZH_FONT)
    ax.set_title("课堂 S-T 分析图", fontproperties=ZH_FONT, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(prop=ZH_FONT)
    plt.tight_layout()
    return fig

def plot_rt_ch_chart(Rt, Ch):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("学生参与度 (Rt)", fontproperties=ZH_FONT)
    ax.set_ylabel("交互有效性 (Ch)", fontproperties=ZH_FONT)
    ax.add_patch(patches.Rectangle((0.7, 0), 0.3, 0.4, color='#f4cccc', alpha=0.5))
    ax.text(0.85, 0.2, '讲授型', fontproperties=ZH_FONT, ha='center')
    ax.plot(Rt, Ch, marker='*', color='red', markersize=12)
    ax.set_title("Rt-Ch 模式分析", fontproperties=ZH_FONT, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_bloom_pie(high, low):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie([high, low], labels=['高阶', '低阶'], autopct='%1.1f%%', startangle=90, textprops={'fontproperties': ZH_FONT})
    ax.set_title("布鲁姆认知层次", fontproperties=ZH_FONT)
    return fig

def plot_four_w_bar(know, app):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(['知识', '应用'], [know, app], color=['#99ff99', '#ffcc99'])
    for tick in ax.get_xticklabels(): tick.set_fontproperties(ZH_FONT)
    ax.set_title("提问维度分布", fontproperties=ZH_FONT)
    return fig

# ==========================================
# 5. UI 主逻辑
# ==========================================
st.set_page_config(page_title="思政课堂分析", layout="wide")
st.title("🎓 思政课堂 AI 评估系统")

user_text = st.text_area("请输入课堂实录文本：", height=150)

if st.button("🚀 开始分析"):
    if not user_text.strip():
        st.warning("⚠️ 请先输入课堂转录文本！")
    else:
        # 创建进度条
        progress_bar = st.progress(0, text="准备开始分析...")
        
        try:
            # --- 模块 1：课堂调控 ---
            progress_bar.progress(10, text="🧠 正在提取课堂调控数据 (1/4)...")
            m1_raw = fetch_evaluation(user_text, SYSTEM_PROMPT_M1)
            m1_data = json.loads(m1_raw.strip().replace('```json', '').replace('```', ''))
            metrics_m1 = calculate_metrics_m1(m1_data)
            
            # --- 模块 2：思维激发 ---
            progress_bar.progress(35, text="🧠 正在分析思维激发维度 (2/4)...")
            m2_raw = fetch_evaluation(user_text, SYSTEM_PROMPT_M2)
            m2_data = json.loads(m2_raw.strip().replace('```json', '').replace('```', ''))
            metrics_m2 = calculate_metrics_m2(m2_data)
            
            # --- 模块 3：核心素养 ---
            progress_bar.progress(60, text="🧠 正在评估学科核心素养 (3/4)...")
            m3_raw = fetch_evaluation(user_text, SYSTEM_PROMPT_M3)
            m3_data = json.loads(m3_raw.strip().replace('```json', '').replace('```', ''))
            metrics_m3 = calculate_metrics_m3(m3_data)
            
            # --- 模块 4：评价反馈 ---
            progress_bar.progress(85, text="🧠 正在诊断评价反馈质量 (4/4)...")
            m4_raw = fetch_evaluation(user_text, SYSTEM_PROMPT_M4)
            m4_data = json.loads(m4_raw.strip().replace('```json', '').replace('```', ''))
            metrics_m4 = calculate_metrics_m4(m4_data)
            
            # --- 综合评分计算 ---
            progress_bar.progress(95, text="✨ 正在生成综合评估报告...")
            overall = calculate_overall_score(metrics_m1, metrics_m2, metrics_m3, metrics_m4)
            
            progress_bar.empty() # 完成后移除进度条
            st.success("✅ 评估完成！")

            # ================== 第一部分：总评报告 (大屏效果) ==================
            st.markdown("---")
            st.markdown("## 🏆 课堂综合评级报告")
            col_res1, col_res2 = st.columns([1, 1.5])
            
            with col_res1:
                st.markdown(f"""
                <div style='text-align: center; border: 2px solid #4CAF50; border-radius: 15px; padding: 30px; background-color: #f9fdf9;'>
                    <h3 style='margin:0; color: #555;'>最终得分</h3>
                    <h1 style='font-size: 60px; color: #4CAF50; margin: 10px 0;'>{overall['total_score']}</h1>
                    <h2 style='color: #2E7D32;'>评级：{overall['final_grade']}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col_res2:
                # 使用 Plotly 绘制雷达图
                st.plotly_chart(plot_overall_radar(
                    overall['m1_norm'], overall['m2_norm'], overall['m3_norm'], overall['m4_norm']
                ), use_container_width=True)

            # ================== 第二部分：详细模块展示 ==================
            st.markdown("### 📊 维度诊断详情")
            tab_m1, tab_m2, tab_m3, tab_m4 = st.tabs(["课堂调控", "思维激发", "核心素养", "评价反馈"])

            with tab_m1:
                st.markdown("#### 🎯 S-T 行为分析与交互频率")
                c1, c2 = st.columns(2)
                with c1: st.pyplot(plot_st_chart(metrics_m1['sequence']))
                with c2: st.pyplot(plot_rt_ch_chart(metrics_m1['Rt'], metrics_m1['Ch']))
                st.info(f"**教学模式判定**：{metrics_m1['teaching_model']}")

            with tab_m2:
                st.markdown("#### 🧠 提问深度与认知层次")
                c3, c4 = st.columns(2)
                with c3: st.pyplot(plot_bloom_pie(metrics_m2['bloom_high'], metrics_m2['bloom_low']))
                with c4: st.pyplot(plot_four_w_bar(metrics_m2['w_know'], metrics_m2['w_app']))
                st.write(f"**教师启发比 (I/D)**: {metrics_m2['id_ratio']:.2f}")

            with tab_m3:
                st.markdown("#### 🌟 核心素养切片打标")
                # 简单列出各维度等级
                for dim in ['political_endorsement', 'scientific_spirit', 'rule_of_law', 'public_participation']:
                    m = metrics_m3[dim]
                    st.write(f"**{dim}**: {m['grade']} (有效切片: {m['total_slices']})")
                with st.expander("🔍 查看原始切片数据"):
                    st.json(metrics_m3['raw_slices'])

            with tab_m4:
                st.markdown("#### 🗣️ 师生评价有效性")
                st.metric("评价反馈率", f"{metrics_m4['feedback_rate']*100:.1f}%")
                st.metric("积极情感率", f"{metrics_m4['positive_rate']*100:.1f}%")
                st.success(f"**专家定性评价**：{metrics_m4['qualitative_evaluation']}")

        except json.JSONDecodeError:
            st.error("❌ AI 返回的数据格式有误，请重试。")
        except Exception as e:
            st.error(f"❌ 分析发生错误：{str(e)}")
