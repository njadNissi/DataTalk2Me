import streamlit as st
import time


SKILLS_INTERESTS = [
    # Original
    "✅ Automatic data preprocessing",
    "📊 Data visualization & plotting",
    "🔍 Descriptive analysis",
    "📈 Regression modeling",
    "🏷️ Classification modeling",
    "🧩 Clustering",
    "🧪 Quick AI prototyping",
    "📚 Education/learning",
    "🔬 Research",
    "💼 Professional work",

    # Data & image & general analysis
    "🧹 Data cleaning & filtering",
    "🖼️ Image conversion & processing",
    "📄 PDF extraction & analysis",
    "📑 Excel/CSV data manipulation",
    "🔎 Exploratory data analysis (EDA)",
    "📐 Statistical analysis & reporting",
    "⏱️ Time series analysis",
    "😊 Sentiment analysis",
    "⚠️ Anomaly & outlier detection",
    "🔗 Correlation & pattern analysis",
    "📋 Automated reporting",
    "🔄 Data transformation & formatting",
    "⚙️ Feature engineering",
    "🤖 Model training & evaluation",
    "🔮 Prediction & forecasting",
    "📝 Document analysis & summarization",
    "✨ Photo enhancement & editing",
    "⚡ Batch file processing",
    "🕸️ Data extraction from files",
    "📏 Data normalization & scaling",
    "📊 Survey & form data analysis",
    "📈 Business intelligence",
    "✅ Data quality checking",
    "📊 Visual report generation",
    "🧪 Machine learning testing",
    "🏷️ Data annotation & labeling",
    "🔗 Data merging & joining",
    "🧪 Data sampling & subset creation",
    "📈 Automated chart generation",
    "💡 Insight & conclusion generation",

    # Sound / audio manipulation
    "🎵 Audio signal processing & filtering",
    "🔄 Sound file format conversion",
    "📶 Audio feature extraction (MFCC, spectral)",
    "🗣️ Speech-to-text transcription & analysis",
    "🔇 Noise reduction & audio cleaning",
    "✂️ Audio segmentation & trimming",
    "📈 Sound visualization (waveforms, spectrograms)",
    "🗣️ Voice analysis & emotion detection",
    "🎚️ Audio mixing & amplitude normalization",
    "📡 Acoustic data analysis & pattern detection",

    # ✅ NEW 20: Programming & Data Structures Learning
    "☕ Java fundamentals & practice",
    "🐍 Python programming learning",
    "🌐 JavaScript & web coding",
    "🧱 C/C++ algorithm development",
    "📱 Kotlin mobile development learning",
    "💻 C# & .NET application learning",
    "🧱 Data structures implementation",
    "📊 Algorithm design & analysis",
    "🔢 Sorting & searching algorithms practice",
    "🌳 Tree & graph algorithms learning",
    "🧩 Object-oriented programming (OOP)",
    "🔗 Linked lists, stacks & queues learning",
    "🗄️ Database & SQL query learning",
    "🔁 Recursion & dynamic programming practice",
    "⚡ Time & space complexity analysis",
    "🧪 Coding exercises & challenges",
    "📝 Pseudocode & logic design",
    "🧠 Memory & performance optimization",
    "🔍 Debugging & code analysis",
    "🤝 Collaborative coding projects"
]


def temp_show(msg:str, type:str, dur:float):
    placeholder = st.empty()

    if type == 'success':
        placeholder.success(msg)
    elif type == 'info':
        placeholder.info(msg)
    elif type == 'error':
        placeholder.error(msg)
    elif type == 'markdown':
        placeholder.markdown(msg)
    else:
        placeholder.write(msg)

    time.sleep(dur)
    placeholder.empty()

    
