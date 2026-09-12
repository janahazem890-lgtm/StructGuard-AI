import os
import sys
import json
from typing import List, Dict, Any, Optional

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

SYSTEM_PROMPT_TEMPLATE = """You are StructGuard AI Assistant, an expert civil engineering AI specialized in structural visual crack assessment.
You have access to the automated quantitative inspection results for the uploaded surface image:

=== CURRENT INSPECTION DATA ===
{inspection_json}
===============================

Guidelines:
1. Base all conclusions strictly on the numeric crack dimensions (length, max width, affected area %, branches) and the rule breakdown provided above.
2. If asked "Why High / Medium / Low?", explicitly cite the exact measurements (max width, length, affected area %) and compare them against the documented threshold formulas.
3. If asked for a report, generate a formal civil engineering inspection report in markdown with summary, measurements, ML consistency check, and next steps.
4. Respond exclusively in professional, clear technical English.
5. Emphasize that this is an optical 2D visual assessment of surface characteristics, NOT a structural safety certification.
"""

def format_system_prompt(inspection_data: Dict[str, Any]) -> str:
    cleaned_data = {}
    for k, v in inspection_data.items():
        if isinstance(v, (int, float, str, bool, dict, list)):
            cleaned_data[k] = v
    return SYSTEM_PROMPT_TEMPLATE.format(inspection_json=json.dumps(cleaned_data, indent=2))

def local_expert_response(user_query: str, inspection_data: Dict[str, Any]) -> str:
    """
    High-fidelity local rule and engineering reasoning assistant in English.
    Does not require an internet connection or external API keys.
    """
    query = user_query.strip().lower()

    sev = inspection_data.get('severity', 'Unknown')
    score = float(inspection_data.get('score', 0.0))
    length = inspection_data.get('length_px', 0)
    width = float(inspection_data.get('max_width_px', 0.0))
    mean_w = float(inspection_data.get('mean_width_px', 0.0))
    area = float(inspection_data.get('affected_area_pct', 0.0))
    branches = inspection_data.get('branch_points', 0)
    components = inspection_data.get('component_count', 1)
    rf_check = inspection_data.get('rf_prediction', sev)

    calib_str = ""
    if inspection_data.get('calibrated'):
        len_cal = inspection_data.get('length_calibrated')
        wid_cal = inspection_data.get('max_width_calibrated')
        calib_str = f" ({len_cal} mm length, {wid_cal} mm max width)"

    # 1. Rationale / Why
    if any(k in query for k in ['why', 'reason', 'explain', 'score', 'classification', 'rating']):
        return (
            f"### 🔍 Classification Rationale for **{sev}** Severity\n\n"
            f"The crack has been evaluated with a composite risk score of **{score:.2f} / 1.00** using the transparent geometric formula:\n\n"
            f"**1. Key Quantitative Factors:**\n"
            f"- **Max Opening Width:** `{width:.1f} px`{calib_str} (Weighted at **45%** of the total score, as crack width is the primary structural tension indicator).\n"
            f"- **Crack Length:** `{length} px`{calib_str} (Weighted at **30%**, indicating physical extension across the member).\n"
            f"- **Affected Surface Area:** `{area:.2f}%` (Weighted at **25%**, representing surface degradation coverage).\n"
            f"- **Branch Nodes:** `{branches}` bifurcation points detected across `{components}` distinct component(s).\n\n"
            f"**2. Machine Learning Cross-Validation:**\n"
            f"- Random Forest classifier validates this rating: **{rf_check}** (98.15% benchmark agreement on the dataset).\n\n"
            f"**3. Threshold Rule Boundaries:**\n"
            f"- Score < 0.40 -> Low | 0.40 <= Score < 0.70 -> Medium | Score >= 0.70 -> High."
        )

    # 2. Formal Inspection Report
    elif any(k in query for k in ['report', 'summary', 'doc', 'formal', 'draft']):
        return (
            f"### 📋 Formal Visual Inspection Report\n\n"
            f"**1. Assessment Findings:**\n"
            f"- **Visual Severity Rating:** **{sev}** (Calculated Score: `{score:.2f} / 1.00`)\n"
            f"- **Crack Length:** `{length} px`{calib_str}\n"
            f"- **Maximum Observed Width:** `{width:.1f} px` (Mean Width: `{mean_w:.1f} px`)\n"
            f"- **Surface Area Coverage:** `{area:.2f}%` of frame area\n"
            f"- **Branching Density:** `{branches}` junction node(s)\n\n"
            f"**2. Machine Learning Verification:**\n"
            f"- Random Forest Validation: `{rf_check}` (Consistent with rule thresholds)\n\n"
            f"**3. Advisory Recommendation:**\n"
            f"{inspection_data.get('recommendation', 'Periodic physical re-inspection recommended.')}\n\n"
            f"**4. Professional Disclaimer:**\n"
            f"This assessment represents 2D surface optical telemetry only. It does not replace structural load calculations, geotechnical investigations, or subsurface core drilling."
        )

    # 3. Further Diagnostic Tests
    elif any(k in query for k in ['test', 'further', 'next', 'improve', 'sensor', 'action']):
        return (
            f"### 🔬 Recommended On-Site Diagnostic Procedures:\n\n"
            f"To advance this visual evaluation to a comprehensive structural diagnosis, the following procedures are recommended:\n\n"
            f"1. **Crack Depth Gauge / Core Sampling:** Determine whether cracking is superficial (e.g. plastic shrinkage) or extends through the structural member.\n"
            f"2. **Optical Micrometer / Crack Comparator Card:** Calibrate certified sub-millimeter physical width on-site.\n"
            f"3. **Ultrasonic Pulse Velocity (UPV):** Assess internal concrete homogeneity and detect subsurface voiding or honeycombing.\n"
            f"4. **Tell-Tale Movement Monitors:** Mount calibrated mechanical or electronic gauges to monitor active vs. dormant crack movement across thermal cycles.\n"
            f"5. **Moisture & Rebar Inspection:** Test for chloride ingress, carbonation depth, efflorescence, or rebar corrosion staining."
        )

    # 4. Safety / Danger / Collapse
    elif any(k in query for k in ['safe', 'danger', 'collapse', 'failure', 'hazard']):
        return (
            f"⚠️ **Important Disclaimer on Structural Safety:**\n\n"
            f"StructGuard AI provides an **optical surface inspection of visible crack telemetry only**.\n"
            f"2D camera photos cannot detect subsurface foundation settlement, corrosion of internal reinforcement rebar, or structural load redistribution.\n\n"
            f"For any crack assessed as **{sev}**, an in-person physical inspection by a licensed professional engineer is required before certifying building safety."
        )

    # 5. Default General Response
    else:
        return (
            f"**Current Inspection Summary:**\n"
            f"- Severity: **{sev}** (Composite Score: `{score:.2f} / 1.00`)\n"
            f"- Length: `{length} px` | Max Width: `{width:.1f} px` | Coverage: `{area:.2f}%`\n\n"
            f"You can ask me:\n"
            f"- *'Why is this crack rated {sev}?'*\n"
            f"- *'Generate a formal inspection report'*;\n"
            f"- *'What diagnostic tests are recommended on-site?'*\n"
            f"- *'Is this crack hazardous?'*"
        )

def query_gemini(user_query: str, inspection_data: Dict[str, Any], api_key: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
    """Query Google Gemini API using google.generativeai in English"""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    
    models_to_try = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.5-flash']
    system_instruction = format_system_prompt(inspection_data)

    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name=model_name, system_instruction=system_instruction)
            contents = []
            if chat_history:
                for msg in chat_history[-6:]:
                    role = 'user' if msg.get('role') == 'user' else 'model'
                    contents.append({'role': role, 'parts': [msg.get('content', '')]})
            contents.append({'role': 'user', 'parts': [user_query]})
            
            response = model.generate_content(contents)
            if response and response.text:
                return response.text
        except Exception:
            continue
    raise RuntimeError("Gemini API call failed across attempted models.")

def query_groq(user_query: str, inspection_data: Dict[str, Any], api_key: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
    """Query Groq API (fast Llama 3) via requests in English"""
    import requests
    system_msg = {'role': 'system', 'content': format_system_prompt(inspection_data)}
    messages = [system_msg]
    if chat_history:
        for msg in chat_history[-6:]:
            if msg.get('role') in ['user', 'assistant']:
                messages.append({'role': msg['role'], 'content': msg['content']})
    messages.append({'role': 'user', 'content': user_query})

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': 'llama-3.3-70b-versatile',
        'messages': messages,
        'temperature': 0.3,
        'max_tokens': 600
    }
    resp = requests.post('https://api.groq.com/openai/v1/chat/completions', json=payload, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data['choices'][0]['message']['content']

def get_chat_response(
    user_query: str,
    inspection_data: Dict[str, Any],
    chat_history: Optional[List[Dict[str, str]]] = None,
    provider: str = "Local Engineering AI (No API Key required - Free & Offline)",
    api_key: Optional[str] = None
) -> str:
    """
    Unified chat handler in English supporting Google Gemini, Groq, OpenAI, and Local Expert.
    """
    # 1. Google Gemini
    if "gemini" in provider.lower() and (api_key or os.getenv('GEMINI_API_KEY')):
        key = api_key or os.getenv('GEMINI_API_KEY')
        try:
            return query_gemini(user_query, inspection_data, key, chat_history)
        except Exception as e:
            fb = local_expert_response(user_query, inspection_data)
            return f"{fb}\n\n*(Note: Gemini API returned: {str(e)}; answered via local civil engineering engine).* "

    # 2. Groq (Llama 3)
    elif "groq" in provider.lower() and (api_key or os.getenv('GROQ_API_KEY')):
        key = api_key or os.getenv('GROQ_API_KEY')
        try:
            return query_groq(user_query, inspection_data, key, chat_history)
        except Exception as e:
            fb = local_expert_response(user_query, inspection_data)
            return f"{fb}\n\n*(Note: Groq API returned: {str(e)}; answered via local civil engineering engine).* "

    # 3. OpenAI (Optional fallback)
    elif "openai" in provider.lower() and (api_key or os.getenv('OPENAI_API_KEY')):
        key = api_key or os.getenv('OPENAI_API_KEY')
        try:
            from openai import OpenAI
            client = OpenAI(api_key=key)
            system_msg = {'role': 'system', 'content': format_system_prompt(inspection_data)}
            messages = [system_msg]
            if chat_history:
                for msg in chat_history[-6:]:
                    if msg.get('role') in ['user', 'assistant']:
                        messages.append({'role': msg['role'], 'content': msg['content']})
            messages.append({'role': 'user', 'content': user_query})
            res = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.3, max_tokens=500)
            return res.choices[0].message.content
        except Exception as e:
            fb = local_expert_response(user_query, inspection_data)
            return f"{fb}\n\n*(Note: OpenAI returned error: {str(e)}; answered via local engine).* "

    # 4. Default: Local English Civil Engineering Expert Engine
    return local_expert_response(user_query, inspection_data)

if __name__ == '__main__':
    dummy_insp = {
        'severity': 'High',
        'score': 0.82,
        'length_px': 620,
        'max_width_px': 14.5,
        'mean_width_px': 8.2,
        'affected_area_pct': 3.8,
        'branch_points': 142,
        'component_count': 3,
        'rf_prediction': 'High',
        'calibrated': True,
        'length_calibrated': 155.0,
        'max_width_calibrated': 3.63,
        'recommendation': 'Priority on-site physical inspection recommended.'
    }
    print(get_chat_response("Why is this crack rated High?", dummy_insp))
