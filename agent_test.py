import os
import time
from typing import TypedDict, List, Annotated, Dict, Union
import operator

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# === 配置 ===
# 请在此处替换你的 API Key，或者确保环境变量 OPENAI_API_KEY 已设置
# os.environ["OPENAI_API_KEY"] = "sk-..."

# 使用 gpt-4o 或 gpt-3.5-turbo (建议使用 gpt-4o 以保证打标和指令理解的准确性)
# llm = ChatOpenAI(model="gpt-4o", temperature=0)

from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def qwen_model(model="qwen-max"):
    model = ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),  # 阿里云颁发的 key
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model=model,  # qwen-plus / qwen-turbo 均可
        temperature=0.5,
        max_retries=2
    )
    print(model)
    return model


llm = qwen_model(model="qwen-max")

# === 0. 定义实体字典 (根据你的Prompt整理) ===
ENTITY_DICT = """
{
    "项目名称": {
        "语义相近的词": ["项目名称", "工程名称", "招标名称"],
        "label": " {{project_name}}"
    },
    "招标人": {
        "语义相近的词": ["招标方", "招标公司", "招标单位", "招标人名称"],
        "label": " {{tenderer_name}}"
    },
    "报价": {
        "语义相近的词": ["报价","投标报价", "报价金", "报价金额", "报价价格"],
        "label": [{"大写报价": "{{uppercase_money}}"}, {"小写报价": "{{lowercase_money}}"}] 
    },
    "交货期": {
        "语义相近的词": ["交货期", "供货周期", "供货期"],
        "label": " {{supply_cycle}}"
    },
    "保证金": {
        "语义相近的词": ["保证金", "保证金额", "投标保证金"],
        "label": [{"大写保证金": "{{uppercase_bond}}"}, {"小写保证金": "{{lowercase_bond}}"}] 
    },
    "投标保证金形式": {
        "语义相近的词": ["保证金形式", "投标保证金形式"],
        "label": " {{bond_form}}"
    }
}
报价金 、保证金金额一般会有大小写之分，从列表中选择对应的大写金额 uppercase， 小写金额：lowercase
"""


# === 1. 定义图的状态 (State) ===
class AgentState(TypedDict):
    original_text: str  # 原始输入文本
    labeling_rules: str  # Planner 生成的规则
    current_labeled_text: str  # Executor 当前的打标结果
    review_feedback: str  # Reviewer 的反馈
    is_approved: bool  # 是否审核通过
    iteration_count: int  # 当前轮次
    messages: Annotated[List[BaseMessage], operator.add]  # 消息历史（可选，用于调试）


# === 2. 定义智能体节点 (Nodes) ===

# --- Planner Agent: 规划者 ---
def planner_agent(state: AgentState):
    print("\n--- [Planner] 正在制定打标方案 ---")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位顶级文本实体打标规划师。"),
        ("human", """
        任务：根据提供的实体字典，分析输入文本，制定具体的打标逻辑说明。

        实体字典：
        {entity_dict}

        输入文本：
        {input_text}

        # 请输出一份给“执行者”看的简明扼要的打标指南，指明在文本中哪些实体关键词附近的空格处需要填充标签。
        #注意：
        1. 不要删除原文中任何一个字符，不能用英文标签替换中文实体，中文实体保留；
        2. 尽可能在中文实体附近的空格处（空白处，且改空白占位符相对较大处）填充对应的一个英文标签（不能一个中文实体有多个英文标签）；
        3. 不需要输出具体打标后的文本，只输出规则。
        """)
    ])

    chain = prompt | llm
    response = chain.invoke({
        "entity_dict": ENTITY_DICT,
        "input_text": state["original_text"]
    })
    print(f"打标方案：", response)
    return {
        "labeling_rules": response.content,
        "iteration_count": state.get("iteration_count", 0)
    }


# --- Executor Agent: 执行者 ---
def executor_agent(state: AgentState):
    print(f"\n--- [Executor] 正在进行第 {state['iteration_count'] + 1} 轮打标 ---")

    # 如果有审核反馈，需要结合反馈进行修改
    feedback_context = ""
    if state.get("review_feedback"):
        feedback_context = f"注意：上一轮打标未通过审核，请根据以下反馈进行修正：\n{state['review_feedback']}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位资深实体打标执行者。
        请根据 Planner 提供的打标逻辑，在输入文本的空格处或者下划线处进行填充打标。

        原则（必须遵循）：
        1. 不要改变原文档的内容以及格式，只需在实体需要的空格处填充对应的标签。
        2. 不能将中文实体用英文标签替换，尽可能在实体前后的空格处（空白处）填充对应的英文标签；
        3. 原文档任何内容不能有任何字符缺失，不能删除任何一个字符。
        4. 标签格式严格为双大括号，例如 {{label}}。
        5. 只需要输出最终打标后的完整文本，不要输出解释。
        """),
        ("human", """
        打标逻辑：
        {rules}

        {feedback_context}

        输入文本：
        {input_text}

        请输出打标后的文本：
        """)
    ])

    chain = prompt | llm
    response = chain.invoke({
        "rules": state["labeling_rules"],
        "feedback_context": feedback_context,
        "input_text": state["original_text"]
    })

    return {
        "current_labeled_text": response.content,
        "iteration_count": state["iteration_count"] + 1
    }


# --- Reviewer Agent: 审核者 ---
def reviewer_agent(state: AgentState):
    print("\n--- [Reviewer] 正在审核打标结果 ---")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是实体检查校对专家。请审查打标内容。

        检查标准：
        1. 是否在实体需要填充的空格/下划线处进行了打标？
        2. 注意打标位置，中文实体前如果有空格空白，英文标签就在实体前空格处打标，中文实体后如果有空格空白，英文标签就在实体后的空格处打标，
        3. 打标的英文字段是否匹配提供的字典？
        4. 标签格式是否严格为: {{label}}？
        5. 打标后原文的内容是否有缺失？，不能将中文实体用英文标签替换，尽可能在中文实体前后的空格处（空白处）填充对应的英文标签；
        6. 打标的中文实体名称不能被英文标签{{label}}替换， 如果替换需要恢复！

        请以JSON格式输出结果，包含字段：
        - "approved": true 或 false
        - "feedback": 具体修改建议（如果通过则为"通过"）
        """),
        ("human", """
        原始实体字典：
        {entity_dict}

        打标结果：
        {labeled_text}

        请给出审核结果：
        """)
    ])

    # 强制 LLM 输出 JSON 结构以便程序解析
    structured_llm = llm.with_structured_output(method="json_mode", schema={
        "type": "object",
        "properties": {
            "approved": {"type": "boolean"},
            "feedback": {"type": "string"}
        },
        "required": ["approved", "feedback"]
    })

    reviewer_chain = prompt | structured_llm
    response = reviewer_chain.invoke({
        "entity_dict": ENTITY_DICT,
        "labeled_text": state["current_labeled_text"]
    })

    print(f"Review 结果: {response['approved']} | 意见: {response['feedback']}")

    return {
        "is_approved": response["approved"],
        "review_feedback": response["feedback"]
    }


# === 3. 定义路由逻辑 (Conditional Edges) ===
def router(state: AgentState):
    # 如果审核通过，或者达到最大轮次（防止死循环），则结束
    if state["is_approved"] or state["iteration_count"] >= 5:
        return "end"
    else:
        return "retry"


# === 4. 构建 LangGraph ===
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("planner", planner_agent)
workflow.add_node("executor", executor_agent)
workflow.add_node("reviewer", reviewer_agent)

# 添加边
workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "reviewer")

# 添加条件边：Reviewer 决定是结束还是重试
workflow.add_conditional_edges(
    "reviewer",
    router,
    {
        "retry": "executor",  # 审核不通过 -> 回到执行者
        "end": END  # 审核通过 -> 结束
    }
)
st = time.time()
app = workflow.compile()

# === 5. 运行测试 ===

input_doc = """
                 投标函
      （招标人名称）：
1．我方已仔细研究了              （项目名称）招标项目招标文件的全部内容，愿意以人民币（含税）（大写）                （小写￥                  元）报价，交货期或供货周期：          ，按合同约定完成             工作。
投标保证金：          万元（大写）（小写￥          ），投标保证金形式：          。
"""

print(f"原始文本:\n{input_doc}")
print("=" * 50)

inputs = {
    "original_text": input_doc,
    "iteration_count": 0,
    "is_approved": False
}

# 执行图
final_state = app.invoke(inputs)

print("=" * 50)
print("=== 最终打标成果 ===")
print(final_state["current_labeled_text"])
print(f"cost time: {time.time()-st}")